import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db import PaperRepository, init_db
from backend.app.main import create_app
from backend.app.services.vector_store import save_embeddings


def make_workspace_tmp(name: str) -> Path:
    """在项目目录中创建 API 测试工作区，避免系统临时目录权限问题。"""

    path = (Path.cwd() / "test_artifacts" / f"{name}-{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def seed_client(workspace: Path) -> tuple[TestClient, Settings]:
    """构造带少量论文的测试应用，专门验证 API 层数据格式。"""

    settings = Settings(project_root=workspace)
    init_db(settings)
    repo = PaperRepository(settings)
    repo.upsert_papers(
        [
            {
                "id": "2401.00001",
                "title": "AI Database Paper",
                "abstract": "A paper that belongs to AI and DB.",
                "authors": "Ada Lovelace",
                "categories": "cs.AI cs.DB",
                "versions": [{"version": "v1", "created": "Mon, 1 Jan 2024 00:00:00 GMT"}],
                "update_date": "2024-01-02",
            },
            {
                "id": "2401.00002",
                "title": "Language Paper",
                "abstract": "A paper that belongs to CL.",
                "authors": "Alan Turing",
                "categories": "cs.CL",
                "versions": [{"version": "v1", "created": "Mon, 5 Feb 2024 00:00:00 GMT"}],
                "update_date": "2024-01-03",
            },
        ],
        source="arxiv_snapshot",
    )
    return TestClient(create_app(settings=settings)), settings


def test_categories_api_returns_configured_categories_with_counts():
    """分类接口应返回 8 个后端配置分类，前端据此动态渲染筛选项。"""

    workspace = make_workspace_tmp("api-categories")
    try:
        client, settings = seed_client(workspace)

        response = client.get("/api/categories")

        assert response.status_code == 200
        items = response.json()["items"]
        by_id = {item["id"]: item["count"] for item in items}
        assert list(by_id) == sorted(settings.target_categories)
        assert by_id["cs.AI"] == 1
        assert by_id["cs.DB"] == 1
        assert by_id["cs.CL"] == 1
        assert by_id["cs.CV"] == 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_papers_api_filters_by_update_date_range():
    """论文列表应能按 update_date 更新日期筛选，供前端论文列表时间过滤使用。"""

    workspace = make_workspace_tmp("api-update-date")
    try:
        client, _settings = seed_client(workspace)

        response = client.get(
            "/api/papers",
            params={
                "time_field": "update_date",
                "date_from": "2024-01-03",
                "date_to": "2024-01-03",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == "2401.00002"
        assert payload["items"][0]["update_date"] == "2024-01-03"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_papers_api_filters_by_created_date_range():
    """论文列表应能按最早版本 created 日期筛选，满足用户可选时间字段的要求。"""

    workspace = make_workspace_tmp("api-created-date")
    try:
        client, _settings = seed_client(workspace)

        response = client.get(
            "/api/papers",
            params={
                "time_field": "created_date",
                "date_from": "2024-02-01",
                "date_to": "2024-02-29",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == "2401.00002"
        assert payload["items"][0]["created_date"] == "2024-02-05"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_papers_api_searches_title_and_abstract_with_bm25_title_priority():
    """论文搜索应使用全文索引和 BM25 排序，标题命中优先于只在摘要中命中的新论文。"""

    workspace = make_workspace_tmp("api-bm25-search")
    try:
        settings = Settings(project_root=workspace)
        init_db(settings)
        repo = PaperRepository(settings)
        repo.upsert_papers(
            [
                {
                    "id": "2401.10001",
                    "title": "Graph Retrieval Framework",
                    "abstract": "A concise system paper.",
                    "authors": "Ada Lovelace",
                    "categories": "cs.IR",
                    "versions": [{"version": "v1", "created": "Mon, 1 Jan 2024 00:00:00 GMT"}],
                    "update_date": "2024-01-01",
                },
                {
                    "id": "2401.10002",
                    "title": "Database Systems Paper",
                    "abstract": "This abstract discusses graph retrieval for indexing.",
                    "authors": "Alan Turing",
                    "categories": "cs.DB",
                    "versions": [{"version": "v1", "created": "Mon, 2 Jan 2024 00:00:00 GMT"}],
                    "update_date": "2024-02-01",
                },
            ],
            source="arxiv_snapshot",
        )
        client = TestClient(create_app(settings=settings))

        response = client.get("/api/papers", params={"query": "graph retrieval", "page_size": 10})

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        assert [item["id"] for item in payload["items"]] == ["2401.10001", "2401.10002"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_papers_api_search_keeps_category_and_time_filters():
    """BM25 搜索仍应能叠加分类和时间筛选，不破坏前端已有筛选组合。"""

    workspace = make_workspace_tmp("api-bm25-filters")
    try:
        client, _settings = seed_client(workspace)

        response = client.get(
            "/api/papers",
            params={
                "query": "paper",
                "category": "cs.CL",
                "time_field": "created_date",
                "date_from": "2024-02-01",
                "date_to": "2024-02-29",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == "2401.00002"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_papers_api_search_ignores_fts_special_characters_safely():
    """包含 FTS 特殊字符的查询不应触发 500，后端会先规范化成安全全文检索词。"""

    workspace = make_workspace_tmp("api-bm25-special")
    try:
        client, _settings = seed_client(workspace)

        response = client.get("/api/papers", params={"query": "AI: (database) + paper?"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] >= 1
        assert any(item["id"] == "2401.00001" for item in payload["items"])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_recommendation_api_applies_same_created_date_filter():
    """推荐结果也应复用同一套时间筛选条件，避免推荐列表展示超出时间范围的论文。"""

    workspace = make_workspace_tmp("api-recommend-date")
    try:
        client, settings = seed_client(workspace)
        paper_ids = ["2401.00001", "2401.00002"]
        save_embeddings(
            settings,
            paper_ids,
            # 两篇论文向量接近，保证如果没有时间过滤，二号论文会进入相似推荐候选。
            np.array([[1.0, 0.0], [0.9, 0.1]], dtype="float32"),
        )
        PaperRepository(settings).mark_embedding_ready(paper_ids)

        response = client.get(
            "/api/papers/2401.00001/recommendations",
            params={
                "top_k": 5,
                "time_field": "created_date",
                "date_from": "2024-02-01",
                "date_to": "2024-02-29",
            },
        )

        assert response.status_code == 200
        items = response.json()["items"]
        assert [item["id"] for item in items] == ["2401.00002"]
        assert items[0]["created_date"] == "2024-02-05"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
