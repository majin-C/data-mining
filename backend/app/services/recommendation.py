from __future__ import annotations

import numpy as np


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """对向量做 L2 归一化，便于用点积计算余弦相似度。"""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def top_k_similar(
    paper_id: str | None,
    paper_ids: list[str],
    embeddings: np.ndarray,
    top_k: int = 10,
    query_vector: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """返回最相似的 Top-K 论文。

    - `paper_id` 不为空时，使用该论文自身 embedding 作为查询，并排除自身。
    - `query_vector` 不为空时，表示用户输入文本已经被编码成向量，不需要排除自身。
    """

    if embeddings.size == 0 or not paper_ids:
        return []
    normalized = _normalize_matrix(embeddings.astype(np.float32))
    if query_vector is None:
        if paper_id not in paper_ids:
            raise KeyError(paper_id)
        query_index = paper_ids.index(paper_id)
        query = normalized[query_index]
    else:
        query = query_vector.astype(np.float32).reshape(1, -1)
        query = _normalize_matrix(query)[0]

    scores = normalized @ query
    candidates: list[tuple[str, float]] = []
    for index, candidate_id in enumerate(paper_ids):
        if paper_id is not None and candidate_id == paper_id:
            continue
        candidates.append((candidate_id, float(scores[index])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [{"id": candidate_id, "score": score} for candidate_id, score in candidates[:top_k]]

