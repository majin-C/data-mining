from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.app.config import Settings


def save_embeddings(settings: Settings, paper_ids: list[str], embeddings: np.ndarray) -> None:
    """保存论文 ID 顺序和 embedding 矩阵。

    论文 ID 文件和 NumPy 矩阵必须一一对应；推荐和聚类都依赖这个顺序。
    """

    settings.ensure_directories()
    np.save(settings.embeddings_path, embeddings.astype(np.float32))
    settings.paper_ids_path.write_text(json.dumps(paper_ids, ensure_ascii=False, indent=2), encoding="utf-8")


def load_embeddings(settings: Settings) -> tuple[list[str], np.ndarray]:
    """读取 embedding 矩阵；文件不存在时返回空矩阵。"""

    if not settings.embeddings_path.exists() or not settings.paper_ids_path.exists():
        return [], np.empty((0, 0), dtype=np.float32)
    paper_ids = json.loads(settings.paper_ids_path.read_text(encoding="utf-8"))
    embeddings = np.load(settings.embeddings_path).astype(np.float32)
    return paper_ids, embeddings


def append_embeddings(
    settings: Settings,
    new_ids: list[str],
    new_embeddings: np.ndarray,
) -> None:
    """向现有 embedding 文件追加新增论文向量。

    手动增量更新只新增论文，不重算历史 embedding，因此追加写入能显著减少更新时间。
    """

    old_ids, old_embeddings = load_embeddings(settings)
    if len(new_ids) == 0:
        return
    if old_embeddings.size == 0:
        save_embeddings(settings, new_ids, new_embeddings)
        return
    merged_ids = [*old_ids, *new_ids]
    merged_embeddings = np.vstack([old_embeddings, new_embeddings.astype(np.float32)])
    save_embeddings(settings, merged_ids, merged_embeddings)
