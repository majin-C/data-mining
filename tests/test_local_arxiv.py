import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from backend.app.config import Settings
from backend.app.datasets.local_arxiv import (
    normalize_arxiv_record,
    rebuild_from_snapshot,
    should_keep_record,
)
from backend.app.db import PaperRepository, init_db


TARGET_CATEGORIES = {"cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.DB", "cs.SE", "cs.DS"}


def write_jsonl(path: Path, records: list[dict]) -> None:
    """写入测试用 JSON Lines 文件，模拟当前本地 arXiv 快照格式。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            # 每条记录独占一行，和真实 1GB 级快照文件保持同一种读取方式。
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_record(arxiv_id: str, categories: str) -> dict:
    """构造最小可入库论文记录，避免每个测试重复展开全部字段。"""

    return {
        "id": arxiv_id,
        "title": "  A Test Paper\n",
        "abstract": "  This paper studies retrieval.\n",
        "authors": "Ada Lovelace",
        "categories": categories,
        "versions": [{"version": "v1", "created": "Mon, 1 Jan 2024 00:00:00 GMT"}],
        "update_date": "2024-01-02",
        "doi": "10.1234/test",
        "journal-ref": "Journal",
        "comments": "10 pages",
    }


def make_workspace_tmp(name: str) -> Path:
    """在项目目录下创建测试隔离目录，避开当前机器的系统临时目录权限问题。"""

    path = (Path.cwd() / "test_artifacts" / f"{name}-{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_settings_uses_all_eight_target_categories():
    """项目配置必须覆盖当前本地快照使用的 8 个 arXiv CS 分区。"""

    workspace = make_workspace_tmp("settings")
    try:
        settings = Settings(project_root=workspace)

        assert set(settings.target_categories) == TARGET_CATEGORIES
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_normalize_arxiv_record_strips_version_and_preserves_snapshot_source():
    """本地快照记录要去掉 arXiv 版本号，并转换成数据库统一字段。"""

    record = make_record("2401.00001v2", "cs.AI cs.IR")

    normalized = normalize_arxiv_record(record)

    assert should_keep_record(record, TARGET_CATEGORIES)
    assert normalized["id"] == "2401.00001"
    assert normalized["title"] == "A Test Paper"
    assert normalized["abstract"] == "This paper studies retrieval."
    assert normalized["journal_ref"] == "Journal"
    assert normalized["source"] == "arxiv_snapshot"


def test_rebuild_from_snapshot_clears_old_state_and_imports_matching_records():
    """全量重建应清空旧数据库、旧向量文件，再导入快照里的全部目标分类论文。"""

    workspace = make_workspace_tmp("rebuild")
    try:
        settings = Settings(project_root=workspace)
        init_db(settings)
        repo = PaperRepository(settings)
        repo.upsert_papers([make_record("old.00001", "cs.AI")], source="legacy")

        settings.ensure_directories()
        np.save(settings.embeddings_path, np.array([[1.0, 0.0]], dtype=np.float32))
        settings.paper_ids_path.write_text(json.dumps(["old.00001"]), encoding="utf-8")

        snapshot = settings.snapshot_path
        write_jsonl(
            snapshot,
            [
                make_record("2401.00001v2", "cs.AI cs.LG"),
                make_record("2401.00002", "math.ST physics.comp-ph"),
                make_record("2401.00003", "cs.DB cs.SE"),
            ],
        )

        inserted = rebuild_from_snapshot(settings, source=snapshot)
        results, total = repo.search_papers(page=1, page_size=10)
        search_results, search_total = repo.search_papers(query="retrieval", page=1, page_size=10)

        assert inserted == 2
        assert total == 2
        assert {paper["id"] for paper in results} == {"2401.00001", "2401.00003"}
        assert search_total == 2
        assert {paper["id"] for paper in search_results} == {"2401.00001", "2401.00003"}
        assert not settings.embeddings_path.exists()
        assert not settings.paper_ids_path.exists()

        # 旧论文不存在时会抛出 KeyError，说明重建没有混入历史数据库内容。
        try:
            repo.get_paper("old.00001")
        except KeyError:
            pass
        else:
            raise AssertionError("旧论文不应在全量重建后继续存在")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
