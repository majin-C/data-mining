import json
import shutil
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.datasets.arxiv_incremental import (
    fetch_incremental_arxiv_records,
    map_arxiv_entry_to_snapshot_record,
    parse_arxiv_feed,
)
from backend.app.db import PaperRepository, init_db
from backend.app.main import create_app
from backend.app.services import sync as sync_service


INITIAL_INCREMENTAL_START = datetime(2026, 5, 30, 23, 59, tzinfo=timezone.utc)
FIXED_WINDOW_END = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)


def make_feed(entries: list[dict]) -> str:
    """生成最小 arXiv Atom XML，测试只关注分页、时间过滤和字段映射。"""

    entry_xml = []
    for entry in entries:
        categories = "".join(f'<category term="{category}" />' for category in entry["categories"])
        entry_xml.append(
            f"""
  <entry>
    <id>http://arxiv.org/abs/{entry["id"]}v1</id>
    <updated>{entry["updated"]}</updated>
    <published>{entry["published"]}</published>
    <title>{entry["title"]}</title>
    <summary>{entry["summary"]}</summary>
    <author><name>{entry["author"]}</name></author>
    {categories}
  </entry>
"""
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
        + "".join(entry_xml)
        + "</feed>"
    )


SAMPLE_FEED = make_feed(
    [
        {
            "id": "2605.31001",
            "updated": "2026-05-31T00:30:00Z",
            "published": "2026-05-31T00:10:00Z",
            "title": "Incremental AI Paper",
            "summary": "This paper is fetched by manual incremental update.",
            "author": "Ada Lovelace",
            "categories": ["cs.AI", "cs.LG"],
        }
    ]
)


class FakeEmbeddingModel:
    """测试用轻量 embedding 模型，避免同步测试加载真实 SPECTER2。"""

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def make_workspace_tmp(name: str) -> Path:
    """在项目目录中创建同步测试工作区，避免系统临时目录权限问题。"""

    path = (Path.cwd() / "test_artifacts" / f"{name}-{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_parse_arxiv_feed_maps_entries_to_snapshot_style_records():
    """arXiv API 的 Atom XML 需要映射成与本地快照一致的字段结构。"""

    entries = parse_arxiv_feed(SAMPLE_FEED)
    record = map_arxiv_entry_to_snapshot_record(entries[0])

    assert record["id"] == "2605.31001"
    assert record["title"] == "Incremental AI Paper"
    assert record["abstract"] == "This paper is fetched by manual incremental update."
    assert record["authors"] == "Ada Lovelace"
    assert record["categories"] == "cs.AI cs.LG"
    assert record["versions"][0]["version"] == "v1"
    assert record["source"] == "arxiv_incremental"


def test_parse_arxiv_feed_filters_by_submitted_window():
    """手动增量只按首次提交时间收录窗口内的新论文。"""

    entries = parse_arxiv_feed(
        SAMPLE_FEED,
        submitted_from=datetime(2026, 5, 31, 0, 0, tzinfo=timezone.utc),
        submitted_until=datetime(2026, 5, 31, 1, 0, tzinfo=timezone.utc),
    )
    assert len(entries) == 1

    outside = parse_arxiv_feed(
        SAMPLE_FEED,
        submitted_from=datetime(2026, 5, 31, 1, 0, tzinfo=timezone.utc),
        submitted_until=datetime(2026, 5, 31, 2, 0, tzinfo=timezone.utc),
    )
    assert outside == []


def test_fetch_incremental_arxiv_records_paginates_until_all_window_records_are_read(monkeypatch):
    """分页抓取应持续请求后续 start，直到当前窗口内没有更多记录。"""

    pages = {
        0: make_feed(
            [
                {
                    "id": "2605.31001",
                    "updated": "2026-05-31T00:10:00Z",
                    "published": "2026-05-31T00:01:00Z",
                    "title": "First Page Paper A",
                    "summary": "page one",
                    "author": "A",
                    "categories": ["cs.AI"],
                },
                {
                    "id": "2605.31002",
                    "updated": "2026-05-31T00:20:00Z",
                    "published": "2026-05-31T00:02:00Z",
                    "title": "First Page Paper B",
                    "summary": "page one",
                    "author": "B",
                    "categories": ["cs.CL"],
                },
            ]
        ),
        2: make_feed(
            [
                {
                    "id": "2605.31003",
                    "updated": "2026-05-31T00:30:00Z",
                    "published": "2026-05-31T00:03:00Z",
                    "title": "Second Page Paper",
                    "summary": "page two",
                    "author": "C",
                    "categories": ["cs.CV"],
                }
            ]
        ),
    }
    requested_starts: list[int] = []

    class FakeResponse:
        """模拟 urllib 响应对象，只实现同步代码实际使用的上下文管理和 read。"""

        def __init__(self, xml_text: str):
            self.xml_text = xml_text

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self.xml_text.encode("utf-8")

    def fake_urlopen(url: str, timeout: int = 30):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = int(query["start"][0])
        requested_starts.append(start)
        return FakeResponse(pages.get(start, make_feed([])))

    monkeypatch.setattr("backend.app.datasets.arxiv_incremental.urllib.request.urlopen", fake_urlopen)

    records = fetch_incremental_arxiv_records(
        {"cs.AI", "cs.CL", "cs.CV"},
        submitted_from=INITIAL_INCREMENTAL_START,
        submitted_until=FIXED_WINDOW_END,
        page_size=2,
    )

    assert requested_starts == [0, 2]
    assert [record["id"] for record in records] == ["2605.31001", "2605.31002", "2605.31003"]


def test_sync_arxiv_incremental_uses_default_baseline_and_updates_state(monkeypatch):
    """首次手动增量应从当前全量数据集截止点开始，并在全部成功后推进状态。"""

    workspace = make_workspace_tmp("incremental-default-baseline")
    try:
        settings = Settings(project_root=workspace)
        init_db(settings)
        captured_window: dict[str, datetime] = {}
        record = {
            "id": "2605.31001",
            "title": "Manual Incremental Paper",
            "abstract": "A new paper from manual incremental update.",
            "authors": "Ada Lovelace",
            "categories": "cs.AI cs.IR",
            "versions": [{"version": "v1", "created": "Sun, 31 May 2026 00:10:00 GMT"}],
            "update_date": "2026-05-31",
            "doi": "",
            "journal_ref": "",
            "comments": "",
            "source": "arxiv_incremental",
        }

        def fake_fetch(categories, submitted_from, submitted_until, **kwargs):
            captured_window["from"] = submitted_from
            captured_window["until"] = submitted_until
            return [record]

        monkeypatch.setattr(sync_service, "fetch_incremental_arxiv_records", fake_fetch)

        result = sync_service.sync_arxiv_incremental(
            settings,
            model=FakeEmbeddingModel(),
            now=FIXED_WINDOW_END,
        )

        repo = PaperRepository(settings)
        assert result["ok"] is True
        assert result["inserted"] == 1
        assert result["embedded"] == 1
        assert captured_window["from"] == INITIAL_INCREMENTAL_START
        assert captured_window["until"] == FIXED_WINDOW_END
        assert repo.get_state(sync_service.LAST_INCREMENTAL_SUCCESS_KEY) == FIXED_WINDOW_END.isoformat()

        lines = settings.incremental_path.read_text(encoding="utf-8").splitlines()
        appended = json.loads(lines[-1])
        assert appended["id"] == "2605.31001"
        assert appended["source"] == "arxiv_incremental"
        assert repo.get_paper("2605.31001")["embedding_ready"] == 1
        assert settings.embeddings_path.exists()
        assert settings.paper_ids_path.exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_sync_arxiv_incremental_uses_saved_last_success(monkeypatch):
    """已有上次成功更新时间时，下一次手动增量应从该时间继续。"""

    workspace = make_workspace_tmp("incremental-saved-baseline")
    try:
        settings = Settings(project_root=workspace)
        init_db(settings)
        repo = PaperRepository(settings)
        saved_start = datetime(2026, 6, 1, 8, 30, tzinfo=timezone.utc)
        repo.set_state(sync_service.LAST_INCREMENTAL_SUCCESS_KEY, saved_start.isoformat())
        captured_window: dict[str, datetime] = {}

        def fake_fetch(categories, submitted_from, submitted_until, **kwargs):
            captured_window["from"] = submitted_from
            captured_window["until"] = submitted_until
            return []

        monkeypatch.setattr(sync_service, "fetch_incremental_arxiv_records", fake_fetch)

        result = sync_service.sync_arxiv_incremental(
            settings,
            model=FakeEmbeddingModel(),
            now=FIXED_WINDOW_END,
        )

        assert result["ok"] is True
        assert captured_window["from"] == saved_start
        assert captured_window["until"] == FIXED_WINDOW_END
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_sync_arxiv_incremental_failure_does_not_advance_last_success(monkeypatch):
    """embedding 失败时不能推进上次成功更新时间，否则下次更新会漏掉本窗口论文。"""

    workspace = make_workspace_tmp("incremental-failure")
    try:
        settings = Settings(project_root=workspace)
        init_db(settings)
        record = {
            "id": "2605.31001",
            "title": "Failure Paper",
            "abstract": "Embedding will fail.",
            "authors": "Ada Lovelace",
            "categories": "cs.AI",
            "versions": [{"version": "v1", "created": "Sun, 31 May 2026 00:10:00 GMT"}],
            "update_date": "2026-05-31",
            "source": "arxiv_incremental",
        }
        monkeypatch.setattr(sync_service, "fetch_incremental_arxiv_records", lambda *args, **kwargs: [record])

        def fail_build(*args, **kwargs):
            raise RuntimeError("embedding failed")

        monkeypatch.setattr(sync_service, "build_missing_embeddings", fail_build)

        result = sync_service.sync_arxiv_incremental(settings, model=FakeEmbeddingModel(), now=FIXED_WINDOW_END)

        repo = PaperRepository(settings)
        assert result["ok"] is False
        assert repo.get_state(sync_service.LAST_INCREMENTAL_SUCCESS_KEY) is None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_incremental_update_api_starts_background_job_and_old_daily_route_is_removed(monkeypatch):
    """前端应通过新的后台任务 API 更新数据，旧 daily 路由不再作为入口存在。"""

    workspace = make_workspace_tmp("incremental-api")
    try:
        settings = Settings(project_root=workspace)
        init_db(settings)
        monkeypatch.setattr(sync_service, "fetch_incremental_arxiv_records", lambda *args, **kwargs: [])
        client = TestClient(create_app(settings=settings))

        start_response = client.post("/api/sync/arxiv-incremental/start")

        assert start_response.status_code == 200
        started = start_response.json()
        assert started["job_id"]
        assert started["status"] in {"running", "completed"}

        status = started
        for _ in range(20):
            status_response = client.get(f"/api/sync/arxiv-incremental/status/{started['job_id']}")
            assert status_response.status_code == 200
            status = status_response.json()
            if status["status"] == "completed":
                break

        assert status["status"] == "completed"
        assert status["result"]["ok"] is True
        assert status["result"]["fetched"] == 0
        assert client.post("/api/sync/arxiv-daily").status_code == 404
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
