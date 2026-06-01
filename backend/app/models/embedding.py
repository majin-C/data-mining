from __future__ import annotations

import hashlib
from typing import Callable, Iterable, Protocol

import numpy as np
import requests

from backend.app.config import Settings
from backend.app.db import PaperRepository
from backend.app.services.vector_store import append_embeddings


SPECTER_BASE_FILES = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]

SPECTER_ADAPTER_FILES = [
    "adapter_config.json",
    "pytorch_adapter.bin",
]


class EmbeddingModel(Protocol):
    """Embedding 模型协议，方便测试中替换真实 SPECTER2 模型。"""

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """把文本列表编码成二维向量矩阵。"""


def paper_to_embedding_text(paper: dict) -> str:
    """把论文标题和摘要拼接成 SPECTER2 输入文本。"""

    return f"{paper.get('title', '')} [SEP] {paper.get('abstract', '')}".strip()


class HashEmbeddingModel:
    """轻量确定性 embedding，用于测试或没有模型依赖时的演示。

    它不是语义模型，但输入相同会得到相同向量，适合单元测试验证数据流。
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeated = (digest * ((self.dimension // len(digest)) + 1))[: self.dimension]
            vector = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
            vector = (vector - 127.5) / 127.5
            vectors.append(vector)
        return np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, self.dimension), dtype=np.float32)


class SpecterEmbeddingModel:
    """SPECTER2 论文 embedding 模型封装。

    该类延迟导入 transformers/adapters/torch，避免运行普通单元测试时强制加载大模型。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None
        self._model = None
        self._device = None

    def _repo_local_dir(self, repo_id: str) -> str:
        """把 Hugging Face 仓库名转换成本地缓存目录名。"""

        return repo_id.replace("/", "__")

    def _download_file_from_mirror(self, repo_id: str, filename: str, target_path) -> None:
        """从 hf-mirror.com 直接下载单个文件。

        当前环境下 `huggingface_hub` 对 hf-mirror 的 HEAD/重定向校验会失败；
        但普通 GET 下载可用。因此这里用 requests 直接下载必要文件，再让
        transformers/adapters 从本地目录加载模型。
        """

        url = f"{self.settings.hf_endpoint.rstrip('/')}/{repo_id}/resolve/main/{filename}"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        # trust_env=False 可以彻底忽略系统代理变量，避免 127.0.0.1:9 这类无效代理影响下载。
        session = requests.Session()
        session.trust_env = False
        with session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temp_path.replace(target_path)

    def _ensure_repo_files(self, repo_id: str, filenames: list[str]):
        """确保指定仓库的必要文件已经下载到本地缓存目录。"""

        local_dir = self.settings.model_cache_dir / self._repo_local_dir(repo_id)
        for filename in filenames:
            target_path = local_dir / filename
            if target_path.exists() and target_path.stat().st_size > 0:
                continue
            print(f"正在从镜像下载 {repo_id}/{filename} ...")
            self._download_file_from_mirror(repo_id, filename, target_path)
        return local_dir

    def _load(self) -> None:
        """加载 SPECTER2 base model 与 proximity adapter。"""

        if self._model is not None:
            return
        self.settings.apply_huggingface_mirror()
        base_dir = self._ensure_repo_files(self.settings.specter_base_model, SPECTER_BASE_FILES)
        adapter_dir = self._ensure_repo_files(self.settings.specter_adapter_model, SPECTER_ADAPTER_FILES)

        # 这些依赖较重，放在函数内部导入，能让没有安装模型依赖的测试仍然运行。
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # transformers/adapters 在 Windows 下对 pathlib.Path 支持不完全，统一传字符串路径。
        self._tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
        self._model = AutoAdapterModel.from_pretrained(str(base_dir))
        self._model.load_adapter(
            str(adapter_dir),
            load_as="specter2",
            set_active=True,
        )
        self._model.to(self._device)
        self._model.eval()

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """批量生成论文语义向量。

        优先使用 GPU；如果本机 CUDA 不可用，torch 会自动走 CPU，保证答辩演示不因硬件差异中断。
        """

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._load()
        import torch

        vectors: list[np.ndarray] = []
        assert self._tokenizer is not None
        assert self._model is not None
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                # SPECTER 系列通常使用 CLS token 表示整篇论文语义。
                batch_vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                vectors.append(batch_vectors.astype(np.float32))
        return np.vstack(vectors).astype(np.float32)


def build_missing_embeddings(
    settings: Settings,
    repo: PaperRepository,
    model: EmbeddingModel | None = None,
    batch_size: int = 16,
    paper_ids: Iterable[str] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> int:
    """为数据库中尚未生成 embedding 的论文追加向量。

    `paper_ids` 不为空时只处理指定论文，避免手动增量更新误处理历史遗留的未生成向量论文。
    `progress_callback` 用于前端后台任务显示真实 embedding 进度；只有所有批次都成功
    编码后才统一追加向量并标记数据库，避免失败时出现半写入状态。
    """

    papers = repo.list_missing_embedding_papers(paper_ids=paper_ids)
    if not papers:
        if progress_callback is not None:
            progress_callback({"completed": 0, "total": 0})
        return 0
    encoder = model or SpecterEmbeddingModel(settings)
    encoded_chunks: list[np.ndarray] = []
    total = len(papers)
    if progress_callback is not None:
        progress_callback({"completed": 0, "total": total})
    for start in range(0, total, batch_size):
        batch_papers = papers[start : start + batch_size]
        texts = [paper_to_embedding_text(paper) for paper in batch_papers]
        encoded_chunks.append(encoder.encode(texts, batch_size=batch_size))
        if progress_callback is not None:
            progress_callback({"completed": min(start + len(batch_papers), total), "total": total})
    embeddings = np.vstack(encoded_chunks).astype(np.float32)
    ready_paper_ids = [paper["id"] for paper in papers]
    append_embeddings(settings, ready_paper_ids, embeddings)
    repo.mark_embedding_ready(ready_paper_ids)
    return len(ready_paper_ids)
