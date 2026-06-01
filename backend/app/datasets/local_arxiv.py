from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from backend.app.config import Settings
from backend.app.db import PaperRepository, connect, init_db


ARXIV_VERSION_RE = re.compile(r"v\d+$")


def clean_text(value: object) -> str:
    """清理 arXiv 文本字段中的多余空白。

    本地快照继承了 arXiv 原始元数据的换行和缩进，标题、摘要、作者等字段在
    入库前统一压缩空白，前端展示和 embedding 输入都会更稳定。
    """

    return " ".join(str(value or "").split())


def strip_arxiv_version(raw_id: object) -> str:
    """去掉 arXiv ID 末尾的版本号。

    当前数据集按无版本 arXiv ID 去重，因此 `2401.00001v2` 和
    `http://arxiv.org/abs/2401.00001v2` 都应入库为 `2401.00001`。
    """

    value = clean_text(raw_id).rsplit("/", maxsplit=1)[-1]
    return ARXIV_VERSION_RE.sub("", value)


def should_keep_record(record: dict, target_categories: set[str] | frozenset[str]) -> bool:
    """判断论文分类是否命中目标 8 个 arXiv CS 分区。"""

    categories = set(clean_text(record.get("categories")).split())
    return bool(categories.intersection(target_categories))


def normalize_arxiv_record(record: dict) -> dict:
    """把本地 arXiv JSONL 记录转换成数据库统一字段。

    字段名称尽量沿用项目现有数据库结构；`journal-ref` 这类原始字段会转换成
    Python/SQLite 更方便使用的 `journal_ref`。
    """

    return {
        "id": strip_arxiv_version(record.get("id")),
        "title": clean_text(record.get("title")),
        "abstract": clean_text(record.get("abstract")),
        "authors": clean_text(record.get("authors")),
        "categories": clean_text(record.get("categories")),
        "versions": record.get("versions") or [],
        "update_date": clean_text(record.get("update_date")),
        "doi": clean_text(record.get("doi")),
        "journal_ref": clean_text(record.get("journal-ref") or record.get("journal_ref")),
        "comments": clean_text(record.get("comments")),
        "source": "arxiv_snapshot",
    }


def iter_arxiv_jsonlines(source: Path) -> Iterator[dict]:
    """逐行读取本地 arXiv JSON Lines 快照。

    快照文件超过 1GB，逐行读取可以避免一次性把全部论文加载进内存。
    """

    if not source.exists():
        raise FileNotFoundError(f"未找到本地 arXiv 快照文件：{source}")
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_and_normalize_records(
    records: Iterable[dict],
    target_categories: set[str] | frozenset[str],
) -> Iterator[dict]:
    """筛选目标分类论文，并跳过缺少核心字段的记录。"""

    for record in records:
        if not should_keep_record(record, target_categories):
            continue
        normalized = normalize_arxiv_record(record)
        # 数据库要求 id/title/abstract 非空；这里提前过滤可以让批量导入过程更稳。
        if normalized["id"] and normalized["title"] and normalized["abstract"]:
            yield normalized


def clear_generated_outputs(settings: Settings) -> None:
    """删除由旧数据集生成的向量和论文 ID 顺序文件。

    全量重建会改变论文集合，旧 embedding 矩阵和 paper_ids 顺序已经不可复用，
    必须删除后再由 `build-embeddings` 单独重新生成。
    """

    for path in (settings.embeddings_path, settings.paper_ids_path):
        if path.exists():
            path.unlink()


def _clear_database_rows(settings: Settings) -> None:
    """清空数据库业务表，保留 SQLite 文件和 schema。

    这样可以避免直接删除数据库文件导致并发连接或目录权限问题，同时实现
    “旧论文、旧聚类、旧统计状态全部清空”的重建语义。
    """

    with connect(settings) as connection:
        # papers_fts 是 papers 的全文倒排索引，重建初始数据集时必须一起清空，
        # 否则旧论文仍可能通过 BM25 搜索结果暴露出来。
        connection.execute("DELETE FROM papers_fts")
        connection.execute("DELETE FROM papers")
        connection.execute("DELETE FROM app_state")


def rebuild_from_snapshot(settings: Settings, source: Path | None = None) -> int:
    """从本地 arXiv 快照全量重建数据库。

    该函数只负责重建论文元数据，不自动生成 embedding；向量生成仍由
    `build-embeddings` 命令单独执行，避免全量导入阶段耗时过长。
    """

    snapshot = source or settings.snapshot_path
    settings.ensure_directories()
    clear_generated_outputs(settings)
    init_db(settings)
    _clear_database_rows(settings)

    repo = PaperRepository(settings)
    inserted = 0
    batch: list[dict] = []
    for paper in filter_and_normalize_records(iter_arxiv_jsonlines(snapshot), settings.target_categories):
        batch.append(paper)
        if len(batch) >= 1000:
            inserted += repo.upsert_papers(batch, source="arxiv_snapshot")
            batch.clear()
    if batch:
        inserted += repo.upsert_papers(batch, source="arxiv_snapshot")
    return inserted
