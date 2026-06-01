from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from backend.app.config import Settings
from backend.app.db import PaperRepository
from backend.app.services.vector_store import load_embeddings


K_EVALUATION_CANDIDATES = tuple(range(8, 31, 2))
K_EVALUATION_SAMPLE_SIZE = 20_000
KMEANS_RANDOM_STATE = 42
CLUSTER_PCA_COMPONENTS = 50
CLUSTER_PREPROCESSING_VERSION = "l2_normalize+pca50:v1"
GPU_PCA_BATCH_SIZE = 16_384
CLUSTER_PLOT_SAMPLE_SIZE = 5_000
ACTIVE_CLUSTER_K_STATE_KEY = "active_cluster_cache_k"
ACTIVE_CLUSTER_SILHOUETTE_STATE_KEY = "active_cluster_cache_silhouette_score"
ACTIVE_CLUSTER_GENERATED_AT_STATE_KEY = "active_cluster_cache_generated_at_utc"
K_EVALUATION_JOBS: dict[str, dict] = {}
K_EVALUATION_JOB_LOCK = threading.Lock()


def _utc_now() -> str:
    """返回 UTC 时间字符串，用于标记 K 值评估缓存的生成时间。"""

    return datetime.now(timezone.utc).isoformat()


def _k_evaluation_cache_path(settings: Settings):
    """返回旧 K 值评估缓存文件路径，仅用于迁移清理。"""

    return settings.processed_data_dir / "k_evaluation_cache.json"


def _delete_legacy_k_evaluation_cache(settings: Settings) -> None:
    """删除旧的独立 K 值评估缓存文件。

    当前前端折线图统一读取 `cluster_cache/manifest.json`，避免同一组 K 的轮廓系数在两个
    文件里各存一份并产生不一致。
    """

    cache_path = _k_evaluation_cache_path(settings)
    try:
        cache_path.unlink()
    except FileNotFoundError:
        return


def _k_evaluation_payload_from_cluster_manifest(settings: Settings) -> dict | None:
    """从全量聚类缓存 manifest 生成前端 K 值折线图数据。"""

    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    if not _cluster_cache_manifest_compatible(manifest):
        return None
    paper_count = int(manifest.get("paper_count") or 0)
    if paper_count <= 0:
        return None
    try:
        _validate_cache_matches_current_embeddings(settings, paper_count)
    except ValueError:
        return None
    items = [
        {
            "k": int(item["k"]),
            "silhouette_score": item.get("silhouette_score"),
            "status": "ok" if item.get("status") == "completed" else str(item.get("status") or "unknown"),
            "message": "",
        }
        for item in manifest.get("items", [])
        if isinstance(item, dict) and item.get("status") == "completed" and "k" in item
    ]
    if not items:
        return None
    return {
        "cached": True,
        "generated_at_utc": manifest.get("generated_at_utc"),
        "preprocessing": manifest.get("preprocessing"),
        "pca_components": manifest.get("pca_components"),
        "actual_pca_components": manifest.get("actual_pca_components"),
        "requested_sample_size": K_EVALUATION_SAMPLE_SIZE,
        "actual_sample_size": int(paper_count),
        "source": "cluster_cache_manifest",
        "items": sorted(items, key=lambda item: item["k"]),
    }


def _cluster_cache_manifest_path(settings: Settings):
    """返回全量 KMeans 预计算缓存的元数据文件路径。"""

    return settings.cluster_cache_dir / "manifest.json"


def _cluster_cache_paper_ids_path(settings: Settings):
    """返回聚类缓存中的论文 ID 顺序文件路径。"""

    return settings.cluster_cache_dir / "paper_ids.json"


def _cluster_cache_coords_path(settings: Settings):
    """返回共享二维 PCA 坐标缓存路径。"""

    return settings.cluster_cache_dir / "coords.npy"


def _cluster_cache_labels_path(settings: Settings, k: int):
    """返回指定 K 的全量 KMeans 标签缓存路径。"""

    return settings.cluster_cache_dir / f"labels_k{int(k)}.npy"


def _read_json_payload(path) -> dict:
    """读取 JSON 文件；文件缺失或损坏时返回空字典，让调用方按无缓存处理。"""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_payload(path, payload: dict) -> None:
    """把 JSON 元数据写入磁盘，统一使用 UTF-8 和缩进格式，便于人工检查。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _array_shape(path) -> tuple[int, ...] | None:
    """只读取 NumPy 文件头部形状，用于快速判断缓存文件是否可复用。"""

    if not path.exists():
        return None
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return None
    return tuple(int(value) for value in array.shape)


def _labels_cache_ready(settings: Settings, k: int, paper_count: int) -> bool:
    """判断指定 K 的标签缓存是否覆盖当前论文数量。"""

    return _array_shape(_cluster_cache_labels_path(settings, k)) == (paper_count,)


def _coords_cache_ready(settings: Settings, paper_count: int) -> bool:
    """判断二维坐标缓存是否覆盖当前论文数量。"""

    return _array_shape(_cluster_cache_coords_path(settings)) == (paper_count, 2)


def _read_cluster_cache_paper_ids(settings: Settings) -> list[str]:
    """读取聚类缓存中的论文 ID 顺序；损坏时抛出明确错误。"""

    path = _cluster_cache_paper_ids_path(settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("尚未生成聚类缓存，请先运行 precompute-clusters。") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("聚类缓存中的 paper_ids.json 损坏，请重新运行 precompute-clusters。") from exc
    if not isinstance(payload, list):
        raise ValueError("聚类缓存中的 paper_ids.json 格式不正确，请重新运行 precompute-clusters。")
    return [str(item) for item in payload]


def _current_embedding_paper_count(settings: Settings) -> int:
    """读取当前向量库论文数量，用于确认缓存仍覆盖当前数据集。"""

    if not settings.paper_ids_path.exists():
        return 0
    try:
        payload = json.loads(settings.paper_ids_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _manifest_items_by_k(manifest: dict) -> dict[int, dict]:
    """把 manifest 中的 items 转成按 K 索引的字典，便于断点续算时覆盖单个 K。"""

    items: dict[int, dict] = {}
    for item in manifest.get("items", []):
        try:
            items[int(item["k"])] = dict(item)
        except (KeyError, TypeError, ValueError):
            continue
    return items


def _cluster_cache_manifest_compatible(manifest: dict, paper_count: int | None = None) -> bool:
    """判断聚类缓存 manifest 是否属于当前 L2+PCA50 预处理流程。

    旧版本缓存虽然文件形状可能仍然匹配，但标签是直接基于原始 embedding 计算的；
    继续复用会让前端展示旧算法结果。因此预处理版本不一致时必须重新预计算。
    """

    if manifest.get("preprocessing") != CLUSTER_PREPROCESSING_VERSION:
        return False
    if int(manifest.get("pca_components") or 0) != CLUSTER_PCA_COMPONENTS:
        return False
    if paper_count is not None and int(manifest.get("paper_count") or 0) != paper_count:
        return False
    return True


def _write_cluster_manifest(settings: Settings, manifest: dict) -> None:
    """写入聚类缓存 manifest。预计算每完成一个 K 都会调用，支持中断后复用。"""

    _write_json_payload(_cluster_cache_manifest_path(settings), manifest)


def _cache_item_for_k(settings: Settings, k: int) -> dict:
    """读取并校验指定 K 的缓存元数据。"""

    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    if not _cluster_cache_manifest_compatible(manifest):
        raise ValueError("聚类缓存使用旧预处理流程，请重新运行 python -m backend.app.cli precompute-clusters。")
    item = _manifest_items_by_k(manifest).get(int(k))
    if not item or item.get("status") != "completed":
        raise ValueError(f"K={k} 尚未预计算，请先运行 python -m backend.app.cli precompute-clusters。")
    paper_count = int(manifest.get("paper_count") or 0)
    if paper_count <= 0:
        raise ValueError("聚类缓存 manifest 缺少论文数量，请重新运行 precompute-clusters。")
    if not _coords_cache_ready(settings, paper_count):
        raise ValueError("聚类二维坐标缓存缺失或数量不匹配，请重新运行 precompute-clusters。")
    if not _labels_cache_ready(settings, int(k), paper_count):
        raise ValueError(f"K={k} 的聚类标签缓存缺失或数量不匹配，请重新运行 precompute-clusters。")
    return {"manifest": manifest, "item": item, "paper_count": paper_count}


def _validate_cache_matches_current_embeddings(settings: Settings, paper_count: int) -> None:
    """确认缓存论文数与当前向量库一致，避免手动增量更新后继续展示旧聚类结果。"""

    current_count = _current_embedding_paper_count(settings)
    if current_count != paper_count:
        raise ValueError(
            f"聚类缓存包含 {paper_count} 篇论文，但当前向量库包含 {current_count} 篇；"
            "请重新运行 precompute-clusters。"
        )


def _copy_job_state(job: dict) -> dict:
    """复制任务状态后再返回给 API，避免调用方误改全局进度字典。"""

    return json.loads(json.dumps(job, ensure_ascii=False))


def _save_job_state(job_id: str, **updates) -> dict:
    """线程安全地更新 K 值评估任务状态。

    后台线程每开始或完成一个候选 K 都会调用这里；前端轮询接口读取同一份状态，
    因此必须加锁，避免读到半更新的数据。
    """

    with K_EVALUATION_JOB_LOCK:
        job = K_EVALUATION_JOBS.setdefault(
            job_id,
            {
                "job_id": job_id,
                "status": "running",
                "completed": 0,
                "total": len(K_EVALUATION_CANDIDATES),
                "current_k": None,
                "cached": False,
                "result": None,
                "error": None,
            },
        )
        job.update(updates)
        return _copy_job_state(job)


def get_k_evaluation_job(job_id: str) -> dict:
    """读取指定 K 值评估任务的当前进度。"""

    with K_EVALUATION_JOB_LOCK:
        job = K_EVALUATION_JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return _copy_job_state(job)


def start_k_evaluation_job(settings: Settings, force: bool = False) -> dict:
    """读取全量聚类缓存中的 K 值评估结果。

    当前 K 值折线图的数据源统一为 `cluster_cache/manifest.json`；`force` 参数保留给前端
    兼容，但不会再触发单独的抽样 KMeans 评估任务。
    """

    job_id = uuid.uuid4().hex
    _delete_legacy_k_evaluation_cache(settings)
    manifest_payload = _k_evaluation_payload_from_cluster_manifest(settings)
    if manifest_payload is not None:
        return _save_job_state(
            job_id,
            status="completed",
            completed=len(manifest_payload["items"]),
            total=len(K_EVALUATION_CANDIDATES),
            current_k=None,
            cached=True,
            result=manifest_payload,
            error=None,
        )

    return _save_job_state(
        job_id,
        status="failed",
        completed=0,
        total=len(K_EVALUATION_CANDIDATES),
        current_k=None,
        cached=False,
        result=None,
        error="尚未生成 KMeans 聚类缓存，请先运行 .\\scripts\\precompute_clusters.ps1。",
    )


def _ensure_embeddings_ready(paper_ids: list[str], embeddings: np.ndarray) -> None:
    """校验 embedding 文件是否可用于聚类。

    聚类、推荐和二维可视化都依赖 `paper_ids.json` 与 `embeddings.npy` 一一对齐；
    如果文件缺失或数量不一致，继续计算会把论文 ID 和向量错配，因此这里提前终止。
    """

    if len(paper_ids) == 0 or embeddings.size == 0:
        raise ValueError("尚未生成 embedding，请先运行 build-embeddings。")
    if len(paper_ids) != len(embeddings):
        raise ValueError("embedding 文件与论文 ID 文件数量不一致，请重新生成 embedding。")


def _sample_indices(total_count: int, sample_size: int, random_state: int = KMEANS_RANDOM_STATE) -> np.ndarray:
    """按固定随机种子抽样 embedding 下标。

    轮廓系数完整计算是 O(n²)；对 615819 篇论文直接计算会非常慢。
    因此 K 值评估固定最多抽样 20000 条，同时用固定种子保证多次评估得到同一批样本。
    """

    if total_count <= sample_size:
        return np.arange(total_count)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(total_count, size=sample_size, replace=False))


def _l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """返回 L2 归一化后的 embedding 副本。

    这里必须复制输入数组，不能在原数组上原地除以范数；原始 `embeddings.npy`
    仍是推荐系统和增量更新依赖的基础向量库，聚类预处理只允许在内存中临时完成。
    """

    normalized = np.array(embeddings, dtype=np.float32, copy=True)
    if normalized.size == 0:
        return normalized
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    non_zero = norms.squeeze(axis=1) > 0
    normalized[non_zero] = normalized[non_zero] / norms[non_zero]
    return normalized


def _cluster_pca_component_count(embeddings: np.ndarray) -> int:
    """计算聚类 PCA 的实际维度，最多 50 维，小样本测试数据会自动降到可用维度。"""

    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        return 0
    return int(min(CLUSTER_PCA_COMPONENTS, embeddings.shape[0], embeddings.shape[1]))


def _cluster_features_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """把原始 embedding 临时转换为 KMeans 使用的 L2+PCA50 聚类特征。"""

    normalized = _l2_normalize_embeddings(embeddings)
    component_count = _cluster_pca_component_count(normalized)
    if component_count == 0:
        return np.empty((len(normalized), 0), dtype=np.float32)
    if normalized.shape[0] == 1:
        return normalized[:, :component_count].astype(np.float32, copy=True)

    from sklearn.decomposition import PCA

    features = PCA(
        n_components=component_count,
        random_state=KMEANS_RANDOM_STATE,
        svd_solver="randomized" if component_count < min(normalized.shape) else "auto",
    ).fit_transform(normalized)
    return np.asarray(features, dtype=np.float32)


def _coords_from_cluster_features(features: np.ndarray) -> np.ndarray:
    """从 PCA50 聚类特征中提取前两维作为前端散点图坐标。"""

    return _pad_to_two_columns(features, len(features))


def _sampled_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int = K_EVALUATION_SAMPLE_SIZE,
    random_state: int = KMEANS_RANDOM_STATE,
) -> float | None:
    """在固定样本上计算轮廓系数，避免全量两两距离计算拖垮课程演示机器。"""

    indices = _sample_indices(len(features), sample_size, random_state=random_state)
    sampled_features = features[indices]
    sampled_labels = labels[indices]
    unique_labels = set(int(label) for label in sampled_labels)
    if len(unique_labels) <= 1 or len(sampled_labels) <= len(unique_labels):
        return None

    from sklearn.metrics import silhouette_score

    return float(silhouette_score(sampled_features, sampled_labels))


def _pad_to_two_columns(coords: np.ndarray, row_count: int) -> np.ndarray:
    """把 PCA 输出统一整理成前端需要的二维坐标矩阵。

    正常论文 embedding 有 768 维，PCA 会直接得到两列；这个函数主要处理测试数据或极端数据：
    例如只有 1 个特征维度时，PCA 只能得到 1 列，需要补一列 0，避免前端散点图读取 y 坐标时报错。
    """

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords.reshape(row_count, 1)
    if coords.shape[1] >= 2:
        return coords[:, :2].astype(np.float32, copy=False)
    padded = np.zeros((row_count, 2), dtype=np.float32)
    if coords.size > 0:
        padded[:, : coords.shape[1]] = coords
    return padded


def _cpu_pca_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """使用 CPU 版 sklearn PCA 作为稳定回退。

    GPU PCA 依赖 PyTorch CUDA；如果用户环境没有可用显卡、CUDA 初始化失败，或显存不足，
    聚类流程仍然应该能完成，所以保留 CPU PCA 兜底。这里不再使用 UMAP，保证降维口径固定为 PCA。
    """

    row_count = len(embeddings)
    if row_count == 0:
        return np.empty((0, 2), dtype=np.float32)
    if row_count == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    from sklearn.decomposition import PCA

    component_count = min(2, row_count, embeddings.shape[1])
    coords = PCA(n_components=component_count, random_state=KMEANS_RANDOM_STATE).fit_transform(embeddings)
    return _pad_to_two_columns(coords, row_count)


def _gpu_pca_to_2d(embeddings: np.ndarray) -> np.ndarray | None:
    """优先在 GPU 上执行 PCA，并在不可用时返回 None 交给 CPU 回退。

    这里没有把 615819x768 的完整矩阵一次性复制到显存，而是分批计算均值、协方差和最终投影：
    - 均值阶段只在 GPU 上累计 768 维向量；
    - 协方差阶段累计 768x768 矩阵；
    - 投影阶段按批次把二维坐标搬回内存。
    这样 RTX 5070 Laptop 这类 8GB/12GB 显存机器也能承受，比直接对全量矩阵做 UMAP 更适合课程演示。
    """

    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    row_count, feature_count = embeddings.shape
    if row_count == 0:
        return np.empty((0, 2), dtype=np.float32)
    if row_count == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    if feature_count == 0:
        return np.zeros((row_count, 2), dtype=np.float32)

    try:
        device = torch.device("cuda")
        source = np.asarray(embeddings, dtype=np.float32)
        batch_size = min(GPU_PCA_BATCH_SIZE, row_count)

        with torch.inference_mode():
            # 第一遍：分批累计均值，避免一次性把完整 embedding 矩阵复制到 GPU。
            total = torch.zeros(feature_count, dtype=torch.float32, device=device)
            for start in range(0, row_count, batch_size):
                end = min(start + batch_size, row_count)
                batch = torch.as_tensor(source[start:end], dtype=torch.float32, device=device)
                total += batch.sum(dim=0)
            mean = total / float(row_count)

            # 第二遍：分批累计协方差矩阵。协方差只有 768x768，显存占用很小。
            covariance = torch.zeros((feature_count, feature_count), dtype=torch.float32, device=device)
            for start in range(0, row_count, batch_size):
                end = min(start + batch_size, row_count)
                batch = torch.as_tensor(source[start:end], dtype=torch.float32, device=device)
                centered = batch - mean
                covariance += centered.T @ centered
            covariance /= float(max(row_count - 1, 1))

            # 第三步：对小矩阵做特征分解，取最大两个特征值对应方向作为 PCA 主成分。
            component_count = min(2, feature_count)
            _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            components = torch.flip(eigenvectors[:, -component_count:], dims=(1,))

            # 第四遍：再次分批投影到二维坐标，并立即搬回 CPU 内存供 SQLite 写回使用。
            coords = np.empty((row_count, component_count), dtype=np.float32)
            for start in range(0, row_count, batch_size):
                end = min(start + batch_size, row_count)
                batch = torch.as_tensor(source[start:end], dtype=torch.float32, device=device)
                projected = (batch - mean) @ components
                coords[start:end] = projected.detach().cpu().numpy()

        return _pad_to_two_columns(coords, row_count)
    except RuntimeError:
        # CUDA OOM 或驱动层运行时错误时释放缓存并回退到 CPU PCA，避免用户一次点击导致服务崩溃。
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def _reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """把高维 embedding 降到二维，供前端聚类散点图展示。

    当前项目固定使用 PCA 口径：优先调用 PyTorch CUDA 版 GPU PCA；如果 GPU 不可用或执行失败，
    自动回退到 sklearn CPU PCA。这里刻意不再导入 UMAP，避免大数据量下 UMAP 耗时过长。
    """

    gpu_coords = _gpu_pca_to_2d(embeddings)
    if gpu_coords is not None:
        return gpu_coords
    return _cpu_pca_to_2d(embeddings)


def _legacy_precompute_cluster_cache(
    settings: Settings,
    candidates: Iterable[int] = K_EVALUATION_CANDIDATES,
    force: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """预计算所有候选 K 的全量 KMeans 标签并保存到本地缓存。

    这一步是课程展示前离线执行的耗时任务。前端“应用该 K”只切换这里生成的缓存，
    不再现场训练 KMeans 或批量更新 SQLite，从而把演示时延迟压到秒级。
    """

    candidate_values = tuple(int(candidate) for candidate in candidates)
    paper_ids, embeddings = load_embeddings(settings)
    _ensure_embeddings_ready(paper_ids, embeddings)

    settings.ensure_directories()
    settings.cluster_cache_dir.mkdir(parents=True, exist_ok=True)

    paper_count = len(paper_ids)
    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    cache_compatible = _cluster_cache_manifest_compatible(manifest, paper_count)
    existing_items = _manifest_items_by_k(manifest) if cache_compatible else {}
    generated_at = _utc_now()
    items_by_k: dict[int, dict] = {
        k: existing_items.get(k, {})
        for k in candidate_values
    }

    cache_paper_ids_path = _cluster_cache_paper_ids_path(settings)
    try:
        cached_paper_count = len(_read_cluster_cache_paper_ids(settings))
    except ValueError:
        cached_paper_count = -1
    if force or not cache_compatible or not cache_paper_ids_path.exists() or cached_paper_count != paper_count:
        cache_paper_ids_path.write_text(json.dumps(paper_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    cluster_features: np.ndarray | None = None
    coords_status = "cached"
    actual_pca_components = int(manifest.get("actual_pca_components") or 0) if cache_compatible else 0
    if force or not cache_compatible or not _coords_cache_ready(settings, paper_count):
        if progress_callback is not None:
            progress_callback(
                {"stage": "preprocessing", "completed": 0, "total": len(candidate_values), "current_k": None}
            )
        cluster_features = _cluster_features_from_embeddings(embeddings)
        actual_pca_components = int(cluster_features.shape[1])
        coords = _coords_from_cluster_features(cluster_features)
        np.save(_cluster_cache_coords_path(settings), coords.astype(np.float32, copy=False))
        coords_status = "generated"

    manifest = {
        "generated_at_utc": generated_at,
        "paper_count": paper_count,
        "candidate_ks": list(candidate_values),
        "random_state": KMEANS_RANDOM_STATE,
        "preprocessing": CLUSTER_PREPROCESSING_VERSION,
        "pca_components": CLUSTER_PCA_COMPONENTS,
        "actual_pca_components": actual_pca_components,
        "paper_ids_file": "paper_ids.json",
        "coords_file": "coords.npy",
        "coords_status": coords_status,
        "items": [items_by_k[k] for k in candidate_values if items_by_k[k]],
    }
    _write_cluster_manifest(settings, manifest)

    from sklearn.cluster import KMeans

    if progress_callback is not None:
        progress_callback({"stage": "kmeans", "completed": 0, "total": len(candidate_values), "current_k": None})

    for index, k in enumerate(candidate_values, start=1):
        if progress_callback is not None:
            progress_callback({"stage": "kmeans", "completed": index - 1, "total": len(candidate_values), "current_k": k})

        labels_path = _cluster_cache_labels_path(settings, k)
        item = items_by_k.get(k, {})
        if (
            not force
            and cache_compatible
            and _labels_cache_ready(settings, k, paper_count)
            and item.get("status") == "completed"
        ):
            item = {**item, "cached": True}
            items_by_k[k] = item
        elif k < 2 or k >= paper_count:
            items_by_k[k] = {
                "k": k,
                "status": "skipped",
                "label_file": labels_path.name,
                "silhouette_score": None,
                "generated_at_utc": generated_at,
                "message": "论文数量不足，无法计算该 K。",
                "cached": False,
            }
        else:
            if cluster_features is None:
                if progress_callback is not None:
                    progress_callback(
                        {"stage": "preprocessing", "completed": index - 1, "total": len(candidate_values), "current_k": k}
                    )
                cluster_features = _cluster_features_from_embeddings(embeddings)
                actual_pca_components = int(cluster_features.shape[1])
            labels = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10).fit_predict(cluster_features)
            labels = np.asarray(labels, dtype=np.int32)
            np.save(labels_path, labels)
            score = _sampled_silhouette_score(cluster_features, labels)
            items_by_k[k] = {
                "k": k,
                "status": "completed",
                "label_file": labels_path.name,
                "silhouette_score": score,
                "generated_at_utc": generated_at,
                "cached": False,
            }

        manifest = {
            **manifest,
            "generated_at_utc": _utc_now(),
            "actual_pca_components": actual_pca_components,
            "items": [items_by_k[item_k] for item_k in candidate_values if items_by_k[item_k]],
        }
        _write_cluster_manifest(settings, manifest)

        if progress_callback is not None:
            progress_callback({"stage": "kmeans", "completed": index, "total": len(candidate_values), "current_k": k})

    return {**manifest, "cache_dir": str(settings.cluster_cache_dir)}


def _legacy_activate_cluster_cache(settings: Settings, repo: PaperRepository, k: int) -> dict:
    """把指定 K 的预计算缓存设置为当前前端展示结果。"""

    payload = _cache_item_for_k(settings, int(k))
    _validate_cache_matches_current_embeddings(settings, int(payload["paper_count"]))
    item = payload["item"]
    generated_at = str(item.get("generated_at_utc") or payload["manifest"].get("generated_at_utc") or "")
    score = item.get("silhouette_score")

    repo.set_state(ACTIVE_CLUSTER_K_STATE_KEY, str(int(k)))
    repo.set_state(ACTIVE_CLUSTER_GENERATED_AT_STATE_KEY, generated_at)
    repo.set_state(ACTIVE_CLUSTER_SILHOUETTE_STATE_KEY, "" if score is None else str(score))

    return {
        "paper_count": int(payload["paper_count"]),
        "cluster_count": int(k),
        "silhouette_score": score,
        "cached": True,
        "generated_at_utc": generated_at,
    }


def _active_cluster_k(repo: PaperRepository) -> int | None:
    """读取当前激活的缓存 K；状态不存在或损坏时返回 None。"""

    value = repo.get_state(ACTIVE_CLUSTER_K_STATE_KEY)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _legacy_active_cluster_summary(settings: Settings, repo: PaperRepository) -> dict | None:
    """读取当前激活缓存的聚类统计，供 `/api/stats` 合并展示。"""

    k = _active_cluster_k(repo)
    if k is None:
        return None
    payload = _cache_item_for_k(settings, k)
    _validate_cache_matches_current_embeddings(settings, int(payload["paper_count"]))
    labels = np.load(_cluster_cache_labels_path(settings, k), mmap_mode="r")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k)
    clusters = [{"cluster_id": index, "count": int(counts[index])} for index in range(k)]
    score = payload["item"].get("silhouette_score")
    return {
        "cluster_count": k,
        "clusters": clusters,
        "silhouette_score": score,
        "generated_at_utc": payload["item"].get("generated_at_utc"),
    }


def _legacy_active_cluster_points(
    settings: Settings,
    repo: PaperRepository,
    sample_size: int = CLUSTER_PLOT_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """从当前激活缓存中读取前端散点图需要的采样点。"""

    k = _active_cluster_k(repo)
    if k is None:
        return []
    payload = _cache_item_for_k(settings, k)
    _validate_cache_matches_current_embeddings(settings, int(payload["paper_count"]))

    paper_ids = _read_cluster_cache_paper_ids(settings)
    coords = np.load(_cluster_cache_coords_path(settings), mmap_mode="r")
    labels = np.load(_cluster_cache_labels_path(settings, k), mmap_mode="r")
    indices = _sample_indices(len(paper_ids), sample_size)
    sampled_ids = [paper_ids[int(index)] for index in indices]
    papers_by_id = repo.get_papers_by_ids(sampled_ids)

    points: list[dict[str, Any]] = []
    for index in indices:
        row_index = int(index)
        paper_id = paper_ids[row_index]
        paper = papers_by_id.get(paper_id, {})
        points.append(
            {
                "id": paper_id,
                "title": paper.get("title", paper_id),
                "categories": paper.get("categories", ""),
                "cluster_id": int(labels[row_index]),
                "x": float(coords[row_index, 0]),
                "y": float(coords[row_index, 1]),
            }
        )
    return points


def cluster_embeddings(settings: Settings, repo: PaperRepository, n_clusters: int = 12) -> dict:
    """执行 KMeans 聚类、二维降维并写回数据库。

    该兼容入口会读取原始 embedding，但只在内存中生成 L2+PCA50 聚类特征；
    不会把归一化结果或 PCA 结果写回 `embeddings.npy`。
    """

    paper_ids, embeddings = load_embeddings(settings)
    _ensure_embeddings_ready(paper_ids, embeddings)
    if len(paper_ids) < n_clusters:
        n_clusters = max(1, len(paper_ids))

    from sklearn.cluster import KMeans

    cluster_features = _cluster_features_from_embeddings(embeddings)
    labels = KMeans(n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE, n_init=10).fit_predict(cluster_features)
    coords = _coords_from_cluster_features(cluster_features)
    results = {
        paper_id: {"cluster_id": int(labels[index]), "x": float(coords[index, 0]), "y": float(coords[index, 1])}
        for index, paper_id in enumerate(paper_ids)
    }
    repo.update_cluster_results(results)

    # 应用 K 时会对全量论文写回聚类结果，但轮廓系数仍固定抽样 20000 条计算；
    # 这样统计卡片和 K 值评估折线图使用同一评价口径，且不会触发全量 O(n²) 距离计算。
    score = _sampled_silhouette_score(cluster_features, labels)
    if score is not None:
        repo.set_state("silhouette_score", str(score))
    return {"paper_count": len(paper_ids), "cluster_count": n_clusters, "silhouette_score": score}


# ---------------------------------------------------------------------------
# 按 arXiv 主分类分组的新版聚类缓存实现
# ---------------------------------------------------------------------------
#
# 上面的旧函数保留给历史入口和单元测试里的底层预处理函数复用；从这里开始，
# 重新定义对外使用的 precompute/evaluate/apply/plot 函数。Python 会以后定义的函数
# 为准，因此 FastAPI 和 CLI 导入到的是下面这套“主分类 + 子聚类”实现。

GROUPED_CLUSTER_CACHE_SCHEMA = "primary_category_kmeans:v1"
ACTIVE_CLUSTER_SELECTION_STATE_KEY = "active_cluster_category_k_selection"
ACTIVE_CLUSTER_SELECTION_GENERATED_AT_KEY = "active_cluster_category_k_generated_at_utc"
DEFAULT_CATEGORY_K = 10
DEFAULT_CATEGORY_ORDER = ("cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.DB", "cs.SE", "cs.DS")


def _target_category_order(settings: Settings) -> list[str]:
    """返回主分类展示顺序。

    Settings.target_categories 是集合类型，本身没有稳定顺序；前端需要 8 个图固定排列，
    因此先按课程数据集的标准顺序输出，再把将来新增但不在默认列表里的分类追加到后面。
    """

    configured = set(settings.target_categories)
    ordered = [category for category in DEFAULT_CATEGORY_ORDER if category in configured]
    ordered.extend(sorted(configured - set(ordered)))
    return ordered


def _primary_category(categories: str | None, target_categories: Iterable[str]) -> str | None:
    """按 arXiv 原始 categories 顺序提取第一命中的目标主分类。

    同一篇论文可能同时属于 `cs.CL cs.LG` 等多个分区。为了保证聚类分组互斥，
    这里不按字母顺序选择，而是尊重数据字段里的原始顺序，取第一个属于目标 8 类的分区。
    """

    target_set = set(target_categories)
    for token in str(categories or "").split():
        if token in target_set:
            return token
    return None


def _category_file_key(category: str) -> str:
    """把 `cs.AI` 这类分类名转换成可读且安全的缓存文件名片段。"""

    return "".join(char if char.isalnum() else "_" for char in category)


def _cluster_cache_category_paper_ids_path(settings: Settings, category: str):
    return settings.cluster_cache_dir / f"paper_ids_{_category_file_key(category)}.json"


def _cluster_cache_category_coords_path(settings: Settings, category: str):
    return settings.cluster_cache_dir / f"coords_{_category_file_key(category)}.npy"


def _cluster_cache_category_labels_path(settings: Settings, category: str, k: int):
    return settings.cluster_cache_dir / f"labels_{_category_file_key(category)}_k{int(k)}.npy"


def _read_category_paper_ids(settings: Settings, category: str) -> list[str]:
    """读取某个主分类内部的论文 ID 顺序。

    每个分类的 labels 和 coords 都按这个顺序排列；读取失败时抛出明确错误，
    方便前端提示用户重新运行预计算脚本。
    """

    path = _cluster_cache_category_paper_ids_path(settings, category)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("尚未生成分组聚类缓存，请先运行 precompute-clusters。") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{category} 的 paper_ids 缓存损坏，请重新运行 precompute-clusters。") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{category} 的 paper_ids 缓存格式不正确，请重新运行 precompute-clusters。")
    return [str(item) for item in payload]


def _category_cache_ready(settings: Settings, category: str, paper_count: int, k: int | None = None) -> bool:
    """检查某个分类的坐标和可选 K 标签文件是否与论文数匹配。"""

    if _array_shape(_cluster_cache_category_coords_path(settings, category)) != (paper_count, 2):
        return False
    if k is None:
        return True
    return _array_shape(_cluster_cache_category_labels_path(settings, category, k)) == (paper_count,)


def _manifest_categories_by_id(manifest: dict) -> dict[str, dict]:
    """把 manifest.categories 转成按分类名索引的字典。"""

    categories: dict[str, dict] = {}
    for item in manifest.get("categories", []):
        if isinstance(item, dict) and item.get("id"):
            categories[str(item["id"])] = dict(item)
    return categories


def _manifest_items_by_category_k(category_manifest: dict) -> dict[int, dict]:
    """把单个分类 manifest 内的 K 结果转成按 K 索引的字典。"""

    items: dict[int, dict] = {}
    for item in category_manifest.get("items", []):
        try:
            items[int(item["k"])] = dict(item)
        except (KeyError, TypeError, ValueError):
            continue
    return items


def _cluster_cache_manifest_compatible(manifest: dict, paper_count: int | None = None) -> bool:
    """判断 manifest 是否属于当前“主分类分组 + L2 + PCA50”缓存版本。

    旧的全局 KMeans manifest 没有 schema 字段，也没有按分类拆分的 categories 字段；
    这里直接判为不兼容，避免前端把旧全局标签误当成 8 个分类内部的子聚类。
    """

    if manifest.get("schema") != GROUPED_CLUSTER_CACHE_SCHEMA:
        return False
    if manifest.get("preprocessing") != CLUSTER_PREPROCESSING_VERSION:
        return False
    if int(manifest.get("pca_components") or 0) != CLUSTER_PCA_COMPONENTS:
        return False
    if not isinstance(manifest.get("categories"), list):
        return False
    if paper_count is not None and int(manifest.get("embedding_count") or manifest.get("paper_count") or 0) != paper_count:
        return False
    return True


def _validate_grouped_cache_matches_current_embeddings(settings: Settings, manifest: dict) -> None:
    """确认分组缓存对应当前 embedding 文件，避免增量更新后继续展示旧标签。"""

    cached_count = int(manifest.get("embedding_count") or manifest.get("paper_count") or 0)
    current_count = _current_embedding_paper_count(settings)
    if cached_count != current_count:
        raise ValueError(
            f"聚类缓存包含 {cached_count} 篇论文，但当前向量库包含 {current_count} 篇；"
            "请重新运行 precompute-clusters。"
        )


def _group_embeddings_by_primary_category(
    settings: Settings,
    repo: PaperRepository,
    paper_ids: list[str],
) -> dict[str, dict[str, list]]:
    """按主分类把 embedding 行号和论文 ID 分组。

    分组只保存索引，不复制 embedding；后续每个分类计算时再按索引切片，这样能减少
    615819 篇论文场景下的内存峰值。
    """

    category_order = _target_category_order(settings)
    grouped: dict[str, dict[str, list]] = {
        category: {"indices": [], "paper_ids": []}
        for category in category_order
    }
    papers_by_id = repo.get_papers_by_ids(paper_ids)
    for index, paper_id in enumerate(paper_ids):
        primary_category = _primary_category(
            papers_by_id.get(paper_id, {}).get("categories", ""),
            category_order,
        )
        if primary_category is None:
            continue
        grouped[primary_category]["indices"].append(index)
        grouped[primary_category]["paper_ids"].append(paper_id)
    return grouped


def _category_item_for_k(settings: Settings, manifest: dict, category: str, k: int) -> dict:
    """读取并校验指定分类、指定 K 的缓存元数据。"""

    category_manifest = _manifest_categories_by_id(manifest).get(category)
    if not category_manifest:
        raise ValueError(f"{category} 尚未生成聚类缓存，请先运行 precompute-clusters。")
    item = _manifest_items_by_category_k(category_manifest).get(int(k))
    if not item or item.get("status") != "completed":
        raise ValueError(f"{category} 的 K={k} 尚未预计算，请先运行 precompute-clusters。")
    paper_count = int(category_manifest.get("paper_count") or 0)
    if not _category_cache_ready(settings, category, paper_count, int(k)):
        raise ValueError(f"{category} 的 K={k} 缓存文件缺失或数量不匹配，请重新运行 precompute-clusters。")
    return {"category": category_manifest, "item": item, "paper_count": paper_count}


def _weighted_silhouette(category_summaries: list[dict]) -> float | None:
    """按分类论文数对各分类 silhouette 做加权平均。"""

    weighted_total = 0.0
    weight = 0
    for item in category_summaries:
        score = item.get("silhouette_score")
        paper_count = int(item.get("paper_count") or 0)
        if score is None or paper_count <= 0:
            continue
        weighted_total += float(score) * paper_count
        weight += paper_count
    if weight == 0:
        return None
    return float(weighted_total / weight)


def _default_selection_from_manifest(manifest: dict) -> dict[str, int]:
    """没有显式应用 K 时，使用每个分类默认 K=10 作为页面初始展示。"""

    selected: dict[str, int] = {}
    for category_manifest in manifest.get("categories", []):
        if not isinstance(category_manifest, dict):
            continue
        category = str(category_manifest.get("id") or "")
        items = _manifest_items_by_category_k(category_manifest)
        if DEFAULT_CATEGORY_K in items and items[DEFAULT_CATEGORY_K].get("status") == "completed":
            selected[category] = DEFAULT_CATEGORY_K
    return selected


def _active_category_selection(settings: Settings, repo: PaperRepository, manifest: dict) -> dict[str, int]:
    """读取当前已应用的“主分类 -> K”映射；没有状态时回退到默认 K=10。"""

    raw_value = repo.get_state(ACTIVE_CLUSTER_SELECTION_STATE_KEY)
    if raw_value:
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            payload = {}
        selected = payload.get("selected", payload)
        if isinstance(selected, dict):
            cleaned: dict[str, int] = {}
            for category, value in selected.items():
                try:
                    cleaned[str(category)] = int(value)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                return cleaned
    return _default_selection_from_manifest(manifest)


def _category_summary_from_selection(settings: Settings, manifest: dict, selected: dict[str, int]) -> list[dict]:
    """根据当前选择的 K 汇总每个主分类的子聚类数量、论文数和 silhouette。"""

    summaries: list[dict] = []
    for category in _target_category_order(settings):
        if category not in selected:
            continue
        k = int(selected[category])
        payload = _category_item_for_k(settings, manifest, category, k)
        category_manifest = payload["category"]
        item = payload["item"]
        labels = np.load(_cluster_cache_category_labels_path(settings, category, k), mmap_mode="r")
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k)
        clusters = [
            {
                "category": category,
                "cluster_id": cluster_index,
                "count": int(counts[cluster_index]),
            }
            for cluster_index in range(k)
        ]
        summaries.append(
            {
                "id": category,
                "paper_count": int(category_manifest.get("paper_count") or 0),
                "current_k": k,
                "cluster_count": k,
                "silhouette_score": item.get("silhouette_score"),
                "generated_at_utc": item.get("generated_at_utc") or manifest.get("generated_at_utc"),
                "clusters": clusters,
            }
        )
    return summaries


def _k_evaluation_payload_from_cluster_manifest(settings: Settings) -> dict | None:
    """从分组 manifest 生成前端 8 张 K 值评估折线图的数据。"""

    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    if not _cluster_cache_manifest_compatible(manifest):
        return None
    try:
        _validate_grouped_cache_matches_current_embeddings(settings, manifest)
    except ValueError:
        return None

    categories = []
    completed_count = 0
    for category in _target_category_order(settings):
        category_manifest = _manifest_categories_by_id(manifest).get(category)
        if not category_manifest:
            continue
        items = []
        for item in category_manifest.get("items", []):
            if not isinstance(item, dict) or item.get("status") != "completed" or "k" not in item:
                continue
            items.append(
                {
                    "k": int(item["k"]),
                    "silhouette_score": item.get("silhouette_score"),
                    "status": "ok",
                    "message": "",
                }
            )
        items.sort(key=lambda item: item["k"])
        completed_count += len(items)
        categories.append(
            {
                "id": category,
                "paper_count": int(category_manifest.get("paper_count") or 0),
                "actual_pca_components": category_manifest.get("actual_pca_components"),
                "items": items,
            }
        )

    if not categories:
        return None
    return {
        "cached": True,
        "generated_at_utc": manifest.get("generated_at_utc"),
        "schema": manifest.get("schema"),
        "preprocessing": manifest.get("preprocessing"),
        "pca_components": manifest.get("pca_components"),
        "candidate_ks": manifest.get("candidate_ks", list(K_EVALUATION_CANDIDATES)),
        "requested_sample_size": K_EVALUATION_SAMPLE_SIZE,
        "actual_sample_size": int(manifest.get("paper_count") or 0),
        "embedding_count": int(manifest.get("embedding_count") or 0),
        "source": "cluster_cache_manifest",
        "completed": completed_count,
        "total": sum(len(category.get("items", [])) for category in categories),
        "categories": categories,
    }


def evaluate_k_candidates(
    settings: Settings,
    candidates: Iterable[int] = K_EVALUATION_CANDIDATES,
    sample_size: int = K_EVALUATION_SAMPLE_SIZE,
    random_state: int = KMEANS_RANDOM_STATE,
    force: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """读取分组聚类 manifest 中已经预计算好的 K 值评估结果。

    前端点击“评估 K 值”时不再现场训练 KMeans；真正耗时的计算由
    `precompute-clusters` 离线完成，这个接口只负责把 8 个分类的折线图数据读出来。
    """

    _delete_legacy_k_evaluation_cache(settings)
    payload = _k_evaluation_payload_from_cluster_manifest(settings)
    if payload is None:
        raise ValueError("尚未生成分组 KMeans 聚类缓存，请先运行 .\\scripts\\precompute_clusters.ps1。")
    if progress_callback is not None:
        progress_callback(
            {
                "completed": int(payload.get("completed") or 0),
                "total": int(payload.get("total") or 0),
                "current_k": None,
                "current_category": None,
            }
        )
    return payload


def start_k_evaluation_job(settings: Settings, force: bool = False) -> dict:
    """创建一个立即完成的 K 值评估任务状态，结果来自分组聚类 manifest。"""

    job_id = uuid.uuid4().hex
    _delete_legacy_k_evaluation_cache(settings)
    manifest_payload = _k_evaluation_payload_from_cluster_manifest(settings)
    if manifest_payload is not None:
        completed = int(manifest_payload.get("completed") or 0)
        total = int(manifest_payload.get("total") or completed)
        return _save_job_state(
            job_id,
            status="completed",
            completed=completed,
            total=total,
            current_k=None,
            current_category=None,
            cached=True,
            result=manifest_payload,
            error=None,
        )

    return _save_job_state(
        job_id,
        status="failed",
        completed=0,
        total=len(K_EVALUATION_CANDIDATES) * len(_target_category_order(settings)),
        current_k=None,
        current_category=None,
        cached=False,
        result=None,
        error="尚未生成分组 KMeans 聚类缓存，请先运行 .\\scripts\\precompute_clusters.ps1。",
    )


def precompute_cluster_cache(
    settings: Settings,
    candidates: Iterable[int] = K_EVALUATION_CANDIDATES,
    force: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """按 arXiv 主分类预计算所有候选 K 的 KMeans 标签并写入本地缓存。

    每个主分类都单独执行 `embedding -> L2 normalize -> PCA50 -> KMeans`。这样得到的是
    “主分类 + 子聚类”的结构，避免全局 KMeans 把不同大方向论文硬塞进同一套簇编号。
    原始 `data/embeddings.npy` 和 `data/paper_ids.json` 只读取，不会被覆盖。
    """

    candidate_values = tuple(int(candidate) for candidate in candidates)
    paper_ids, embeddings = load_embeddings(settings)
    _ensure_embeddings_ready(paper_ids, embeddings)

    settings.ensure_directories()
    settings.cluster_cache_dir.mkdir(parents=True, exist_ok=True)
    repo = PaperRepository(settings)
    grouped = _group_embeddings_by_primary_category(settings, repo, paper_ids)
    embedding_count = len(paper_ids)
    assigned_count = sum(len(group["paper_ids"]) for group in grouped.values())

    old_manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    cache_compatible = _cluster_cache_manifest_compatible(old_manifest, embedding_count)
    old_categories = _manifest_categories_by_id(old_manifest) if cache_compatible else {}
    generated_at = _utc_now()
    total_steps = len(candidate_values) * len(grouped)
    completed_steps = 0
    category_manifests: list[dict] = []

    from sklearn.cluster import KMeans

    for category in _target_category_order(settings):
        group = grouped.get(category, {"indices": [], "paper_ids": []})
        indices = [int(index) for index in group["indices"]]
        category_paper_ids = [str(paper_id) for paper_id in group["paper_ids"]]
        category_count = len(category_paper_ids)
        old_category = old_categories.get(category, {})
        old_items = _manifest_items_by_category_k(old_category) if cache_compatible else {}
        items_by_k: dict[int, dict] = {k: old_items.get(k, {}) for k in candidate_values}
        paper_ids_path = _cluster_cache_category_paper_ids_path(settings, category)

        if force or not cache_compatible or not paper_ids_path.exists():
            paper_ids_path.write_text(json.dumps(category_paper_ids, ensure_ascii=False, indent=2), encoding="utf-8")

        features: np.ndarray | None = None
        actual_pca_components = int(old_category.get("actual_pca_components") or 0) if cache_compatible else 0
        coords_status = "cached"
        if category_count > 0 and (
            force
            or not cache_compatible
            or not _category_cache_ready(settings, category, category_count)
        ):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "preprocessing",
                        "current_category": category,
                        "current_k": None,
                        "completed": completed_steps,
                        "total": total_steps,
                    }
                )
            features = _cluster_features_from_embeddings(embeddings[indices])
            actual_pca_components = int(features.shape[1])
            coords = _coords_from_cluster_features(features)
            np.save(_cluster_cache_category_coords_path(settings, category), coords.astype(np.float32, copy=False))
            coords_status = "generated"
        elif category_count == 0:
            actual_pca_components = 0
            coords_status = "empty"

        category_manifest = {
            "id": category,
            "paper_count": category_count,
            "candidate_ks": list(candidate_values),
            "paper_ids_file": _cluster_cache_category_paper_ids_path(settings, category).name,
            "coords_file": _cluster_cache_category_coords_path(settings, category).name,
            "coords_status": coords_status,
            "actual_pca_components": actual_pca_components,
            "items": [items_by_k[k] for k in candidate_values if items_by_k[k]],
        }

        for k in candidate_values:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "kmeans",
                        "current_category": category,
                        "current_k": k,
                        "completed": completed_steps,
                        "total": total_steps,
                    }
                )

            labels_path = _cluster_cache_category_labels_path(settings, category, k)
            item = items_by_k.get(k, {})
            if (
                not force
                and cache_compatible
                and item.get("status") == "completed"
                and _category_cache_ready(settings, category, category_count, k)
            ):
                items_by_k[k] = {**item, "cached": True}
            elif k < 2 or k >= category_count:
                items_by_k[k] = {
                    "k": k,
                    "status": "skipped",
                    "label_file": labels_path.name,
                    "silhouette_score": None,
                    "generated_at_utc": generated_at,
                    "message": "该分类论文数量不足，无法计算该 K。",
                    "cached": False,
                }
            else:
                if features is None:
                    features = _cluster_features_from_embeddings(embeddings[indices])
                    actual_pca_components = int(features.shape[1])
                    if not _category_cache_ready(settings, category, category_count):
                        coords = _coords_from_cluster_features(features)
                        np.save(_cluster_cache_category_coords_path(settings, category), coords.astype(np.float32, copy=False))
                labels = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10).fit_predict(features)
                labels = np.asarray(labels, dtype=np.int32)
                np.save(labels_path, labels)
                score = _sampled_silhouette_score(features, labels)
                items_by_k[k] = {
                    "k": k,
                    "status": "completed",
                    "label_file": labels_path.name,
                    "silhouette_score": score,
                    "generated_at_utc": generated_at,
                    "cached": False,
                }

            completed_steps += 1
            category_manifest = {
                **category_manifest,
                "actual_pca_components": actual_pca_components,
                "items": [items_by_k[item_k] for item_k in candidate_values if items_by_k[item_k]],
            }

            partial_categories = [item for item in category_manifests if item.get("id") != category]
            partial_categories.append(category_manifest)
            partial_categories.sort(key=lambda item: _target_category_order(settings).index(item["id"]))
            _write_cluster_manifest(
                settings,
                {
                    "schema": GROUPED_CLUSTER_CACHE_SCHEMA,
                    "generated_at_utc": _utc_now(),
                    "paper_count": assigned_count,
                    "embedding_count": embedding_count,
                    "candidate_ks": list(candidate_values),
                    "random_state": KMEANS_RANDOM_STATE,
                    "preprocessing": CLUSTER_PREPROCESSING_VERSION,
                    "pca_components": CLUSTER_PCA_COMPONENTS,
                    "grouping": "primary_category:first_target_match",
                    "default_k": DEFAULT_CATEGORY_K,
                    "categories": partial_categories,
                },
            )

            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "kmeans",
                        "current_category": category,
                        "current_k": k,
                        "completed": completed_steps,
                        "total": total_steps,
                    }
                )

        category_manifests.append(category_manifest)

    manifest = {
        "schema": GROUPED_CLUSTER_CACHE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "paper_count": assigned_count,
        "embedding_count": embedding_count,
        "candidate_ks": list(candidate_values),
        "random_state": KMEANS_RANDOM_STATE,
        "preprocessing": CLUSTER_PREPROCESSING_VERSION,
        "pca_components": CLUSTER_PCA_COMPONENTS,
        "grouping": "primary_category:first_target_match",
        "default_k": DEFAULT_CATEGORY_K,
        "categories": category_manifests,
    }
    _write_cluster_manifest(settings, manifest)
    return {**manifest, "cache_dir": str(settings.cluster_cache_dir)}


def activate_cluster_cache(settings: Settings, repo: PaperRepository, selected: dict[str, int]) -> dict:
    """应用前端提交的“主分类 -> K”映射。

    这个操作只写入 app_state，表示当前页面应读取哪些本地标签文件；不会重新计算 KMeans，
    也不会把 61 万行聚类标签批量写回 SQLite。
    """

    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    if not _cluster_cache_manifest_compatible(manifest):
        raise ValueError("尚未生成分组 KMeans 聚类缓存，请先运行 precompute-clusters。")
    _validate_grouped_cache_matches_current_embeddings(settings, manifest)

    selected_map = dict(_default_selection_from_manifest(manifest))
    for category, value in selected.items():
        if category not in _manifest_categories_by_id(manifest):
            continue
        selected_map[str(category)] = int(value)
    if not selected_map:
        raise ValueError("没有可应用的分类 K 值，请先运行 precompute-clusters。")

    summaries = _category_summary_from_selection(settings, manifest, selected_map)
    if not summaries:
        raise ValueError("所选 K 值没有对应的本地缓存，请先运行 precompute-clusters。")

    generated_at = _utc_now()
    repo.set_state(
        ACTIVE_CLUSTER_SELECTION_STATE_KEY,
        json.dumps({"selected": selected_map}, ensure_ascii=False, sort_keys=True),
    )
    repo.set_state(ACTIVE_CLUSTER_SELECTION_GENERATED_AT_KEY, generated_at)

    cluster_count = sum(int(item["cluster_count"]) for item in summaries)
    paper_count = sum(int(item["paper_count"]) for item in summaries)
    return {
        "paper_count": paper_count,
        "cluster_count": cluster_count,
        "silhouette_score": _weighted_silhouette(summaries),
        "cached": True,
        "generated_at_utc": generated_at,
        "selected": selected_map,
        "categories": summaries,
    }


def active_cluster_summary(settings: Settings, repo: PaperRepository) -> dict | None:
    """返回当前已应用或默认 K=10 的分组聚类统计。"""

    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    if not _cluster_cache_manifest_compatible(manifest):
        return None
    _validate_grouped_cache_matches_current_embeddings(settings, manifest)
    selected = _active_category_selection(settings, repo, manifest)
    summaries = _category_summary_from_selection(settings, manifest, selected)
    if not summaries:
        return None
    clusters = [cluster for category in summaries for cluster in category["clusters"]]
    return {
        "cluster_count": sum(int(item["cluster_count"]) for item in summaries),
        "clusters": clusters,
        "silhouette_score": _weighted_silhouette(summaries),
        "generated_at_utc": repo.get_state(ACTIVE_CLUSTER_SELECTION_GENERATED_AT_KEY) or manifest.get("generated_at_utc"),
        "cluster_categories": summaries,
        "selected": selected,
    }


def active_cluster_points(
    settings: Settings,
    repo: PaperRepository,
    sample_size: int = CLUSTER_PLOT_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """返回 8 个主分类各自的聚类散点图数据。"""

    manifest = _read_json_payload(_cluster_cache_manifest_path(settings))
    if not _cluster_cache_manifest_compatible(manifest):
        return []
    _validate_grouped_cache_matches_current_embeddings(settings, manifest)
    selected = _active_category_selection(settings, repo, manifest)
    groups: list[dict[str, Any]] = []

    for category in _target_category_order(settings):
        if category not in selected:
            continue
        k = int(selected[category])
        payload = _category_item_for_k(settings, manifest, category, k)
        category_manifest = payload["category"]
        paper_ids = _read_category_paper_ids(settings, category)
        coords = np.load(_cluster_cache_category_coords_path(settings, category), mmap_mode="r")
        labels = np.load(_cluster_cache_category_labels_path(settings, category, k), mmap_mode="r")
        indices = _sample_indices(len(paper_ids), sample_size)
        sampled_ids = [paper_ids[int(index)] for index in indices]
        papers_by_id = repo.get_papers_by_ids(sampled_ids)

        points: list[dict[str, Any]] = []
        for index in indices:
            row_index = int(index)
            paper_id = paper_ids[row_index]
            paper = papers_by_id.get(paper_id, {})
            subcluster_id = int(labels[row_index])
            points.append(
                {
                    "id": paper_id,
                    "title": paper.get("title", paper_id),
                    "categories": paper.get("categories", ""),
                    "primary_category": category,
                    "subcluster_id": subcluster_id,
                    "cluster_id": subcluster_id,
                    "x": float(coords[row_index, 0]),
                    "y": float(coords[row_index, 1]),
                }
            )

        groups.append(
            {
                "id": category,
                "paper_count": int(category_manifest.get("paper_count") or 0),
                "shown_count": len(points),
                "k": k,
                "silhouette_score": payload["item"].get("silhouette_score"),
                "points": points,
            }
        )

    return groups
