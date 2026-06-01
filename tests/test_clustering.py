import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db import PaperRepository, init_db
from backend.app.main import create_app
from backend.app.services import clustering
from backend.app.services.clustering import evaluate_k_candidates, precompute_cluster_cache
from backend.app.services.vector_store import save_embeddings


TEST_CATEGORIES = ("cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.DB", "cs.SE", "cs.DS")


def make_workspace_tmp(name: str) -> Path:
    """在项目目录下创建测试工作区。

    测试会写 SQLite、embedding 和聚类缓存文件；放在项目 test_artifacts 下可以避开
    系统临时目录权限差异，并在测试结束后统一清理。
    """

    path = (Path.cwd() / "test_artifacts" / f"{name}-{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def seed_cluster_client(workspace: Path, paper_count: int = 64) -> tuple[TestClient, Settings]:
    """构造覆盖 8 个 arXiv 主分类的论文和 embedding。

    每篇论文只写一个主分类，方便测试分组数量和每组标签文件是否一一对应。
    embedding 使用固定随机种子，保证 KMeans、PCA 和 silhouette 在测试中可重复。
    """

    settings = Settings(project_root=workspace)
    init_db(settings)
    repo = PaperRepository(settings)
    paper_ids = [f"2401.{index:05d}" for index in range(paper_count)]
    repo.upsert_papers(
        [
            {
                "id": paper_id,
                "title": f"Cluster Test Paper {index}",
                "abstract": f"Synthetic abstract for grouped cluster evaluation {index}.",
                "authors": "Test Author",
                "categories": TEST_CATEGORIES[index % len(TEST_CATEGORIES)],
                "versions": [],
                "update_date": "2024-01-01",
            }
            for index, paper_id in enumerate(paper_ids)
        ],
        source="arxiv_snapshot",
    )

    rng = np.random.default_rng(20260530)
    embeddings = rng.normal(size=(paper_count, 6)).astype(np.float32)
    save_embeddings(settings, paper_ids, embeddings)
    repo.mark_embedding_ready(paper_ids)
    return TestClient(create_app(settings=settings)), settings


def test_primary_category_uses_first_target_category():
    """多分类论文应按 categories 字段里的第一个目标分类归组。"""

    ordered = ["cs.AI", "cs.CL", "cs.CV"]

    assert clustering._primary_category("cs.CL cs.AI", ordered) == "cs.CL"
    assert clustering._primary_category("math.CO cs.CV cs.AI", ordered) == "cs.CV"
    assert clustering._primary_category("math.CO stat.ML", ordered) is None


def test_cluster_preprocessing_l2_normalizes_without_mutating_input():
    """聚类预处理只能处理副本，不能覆盖原始 embedding 数组。"""

    embeddings = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    original = embeddings.copy()

    normalized = clustering._l2_normalize_embeddings(embeddings)

    np.testing.assert_allclose(embeddings, original)
    np.testing.assert_allclose(normalized[0], np.array([0.6, 0.8, 0.0], dtype=np.float32))
    np.testing.assert_allclose(normalized[1], np.zeros(3, dtype=np.float32))


def test_cluster_features_use_pca50_or_smaller_dimension():
    """KMeans 输入特征应来自 L2 归一化后的 PCA，维度最多 50。"""

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(80, 64)).astype(np.float32)

    features = clustering._cluster_features_from_embeddings(embeddings)

    assert features.shape == (80, 50)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_grouped_precompute_writes_category_files_and_manifest():
    """预计算应为每个主分类分别写入 paper_ids、coords 和 labels 文件。"""

    workspace = make_workspace_tmp("cluster-grouped-precompute")
    try:
        _client, settings = seed_cluster_client(workspace)

        result = precompute_cluster_cache(settings, candidates=[2, 3], force=True)

        cache_dir = settings.cluster_cache_dir
        assert Path(result["cache_dir"]) == cache_dir
        assert result["schema"] == clustering.GROUPED_CLUSTER_CACHE_SCHEMA
        assert result["preprocessing"] == clustering.CLUSTER_PREPROCESSING_VERSION
        assert result["paper_count"] == 64
        assert result["embedding_count"] == 64
        assert [item["id"] for item in result["categories"]] == list(TEST_CATEGORIES)

        for category in TEST_CATEGORIES:
            file_key = category.replace(".", "_")
            assert (cache_dir / f"paper_ids_{file_key}.json").exists()
            assert np.load(cache_dir / f"coords_{file_key}.npy").shape == (8, 2)
            assert np.load(cache_dir / f"labels_{file_key}_k2.npy").shape == (8,)
            assert np.load(cache_dir / f"labels_{file_key}_k3.npy").shape == (8,)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_evaluate_k_api_returns_grouped_manifest_data_and_ignores_legacy_cache():
    """K 值评估接口应直接读取分组 manifest，不再读取 k_evaluation_cache.json。"""

    workspace = make_workspace_tmp("cluster-grouped-evaluate")
    try:
        client, settings = seed_cluster_client(workspace)
        precompute_cluster_cache(settings, candidates=[2, 3], force=True)

        legacy_cache = settings.processed_data_dir / "k_evaluation_cache.json"
        legacy_cache.write_text(
            json.dumps({"items": [{"k": 99, "silhouette_score": 1.0}]}, ensure_ascii=False),
            encoding="utf-8",
        )

        response = client.post("/api/clusters/evaluate-k")

        assert response.status_code == 200
        payload = response.json()
        assert payload["cached"] is True
        assert payload["source"] == "cluster_cache_manifest"
        assert payload["schema"] == clustering.GROUPED_CLUSTER_CACHE_SCHEMA
        assert payload["actual_sample_size"] == 64
        assert [item["id"] for item in payload["categories"]] == list(TEST_CATEGORIES)
        for category in payload["categories"]:
            assert [item["k"] for item in category["items"]] == [2, 3]
            assert all(isinstance(item["silhouette_score"], float) for item in category["items"])
        assert not legacy_cache.exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_evaluate_k_candidates_reports_grouped_progress_as_completed():
    """读取 manifest 时应上报 8 个分类乘以候选 K 的完成进度。"""

    workspace = make_workspace_tmp("cluster-grouped-progress")
    try:
        _client, settings = seed_cluster_client(workspace)
        precompute_cluster_cache(settings, candidates=[2, 3], force=True)
        events: list[dict] = []

        payload = evaluate_k_candidates(settings, candidates=[2, 3], progress_callback=events.append)

        assert payload["completed"] == 16
        assert payload["total"] == 16
        assert events == [{"completed": 16, "total": 16, "current_k": None, "current_category": None}]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_apply_k_api_uses_grouped_cache_and_does_not_write_sqlite_clusters():
    """应用 K 只切换本地分组缓存，不应批量写回 papers.cluster_id/x/y。"""

    workspace = make_workspace_tmp("cluster-grouped-apply")
    try:
        client, settings = seed_cluster_client(workspace)
        precompute_cluster_cache(settings, candidates=[2, 3], force=True)
        selected = {category: 2 for category in TEST_CATEGORIES}

        response = client.post("/api/clusters/apply-k", json={"selected": selected})

        assert response.status_code == 200
        payload = response.json()
        assert payload["selected"] == selected
        assert payload["cluster_count"] == 16
        assert payload["paper_count"] == 64
        assert isinstance(payload["silhouette_score"], float)

        stats = client.get("/api/stats").json()
        assert stats["cluster_count"] == 16
        assert stats["selected"] == selected
        assert len(stats["cluster_categories"]) == 8
        assert sum(item["paper_count"] for item in stats["cluster_categories"]) == 64

        plot = client.get("/api/clusters/plot").json()
        assert len(plot["groups"]) == 8
        assert {group["id"] for group in plot["groups"]} == set(TEST_CATEGORIES)
        for group in plot["groups"]:
            assert group["k"] == 2
            assert group["paper_count"] == 8
            assert len(group["points"]) == 8
            assert {point["primary_category"] for point in group["points"]} == {group["id"]}
            assert {point["subcluster_id"] for point in group["points"]}.issubset({0, 1})

        repo = PaperRepository(settings)
        assert repo.cluster_points() == []
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_apply_k_requires_grouped_precomputed_cache():
    """没有本地分组 KMeans 缓存时，应用 K 应返回明确错误。"""

    workspace = make_workspace_tmp("cluster-grouped-missing-cache")
    try:
        client, _settings = seed_cluster_client(workspace)

        response = client.post("/api/clusters/apply-k", json={"selected": {"cs.AI": 2}})

        assert response.status_code == 400
        assert "precompute-clusters" in response.json()["detail"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_old_global_cluster_manifest_is_not_reused():
    """旧全局 manifest 不应被新分组逻辑误认为可用。"""

    workspace = make_workspace_tmp("cluster-grouped-old-cache")
    try:
        client, settings = seed_cluster_client(workspace)
        settings.cluster_cache_dir.mkdir(parents=True, exist_ok=True)
        (settings.cluster_cache_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "paper_count": 64,
                    "preprocessing": clustering.CLUSTER_PREPROCESSING_VERSION,
                    "pca_components": 50,
                    "items": [{"k": 10, "status": "completed", "silhouette_score": 0.1}],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        response = client.post("/api/clusters/apply-k", json={"selected": {"cs.AI": 10}})

        assert response.status_code == 400
        assert "precompute-clusters" in response.json()["detail"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_precompute_and_evaluate_do_not_overwrite_embedding_files():
    """分组预计算和评估都只能读取原始 embedding，不能覆盖向量库文件。"""

    workspace = make_workspace_tmp("cluster-grouped-embedding-readonly")
    try:
        _client, settings = seed_cluster_client(workspace)
        before_embeddings = np.load(settings.embeddings_path).copy()
        before_ids = settings.paper_ids_path.read_text(encoding="utf-8")

        precompute_cluster_cache(settings, candidates=[2], force=True)
        evaluate_k_candidates(settings, candidates=[2], sample_size=20, force=True)

        after_embeddings = np.load(settings.embeddings_path)
        after_ids = settings.paper_ids_path.read_text(encoding="utf-8")
        np.testing.assert_allclose(after_embeddings, before_embeddings)
        assert after_ids == before_ids
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
