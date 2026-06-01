import json
import shutil
import sys
import uuid
from pathlib import Path

from backend.app import cli
from backend.app.config import Settings
from backend.app.db import PaperRepository


def make_workspace_tmp(name: str) -> Path:
    """在项目目录中创建 CLI 测试工作区，避免依赖系统临时目录。"""

    path = (Path.cwd() / "test_artifacts" / f"{name}-{uuid.uuid4().hex}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_jsonl(path: Path, records: list[dict]) -> None:
    """写入最小本地 arXiv 快照，供 CLI 默认导入路径使用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_import_arxiv_snapshot_cli_rebuilds_from_default_snapshot(monkeypatch):
    """CLI 应提供 import-arxiv-snapshot，并默认从 Settings.snapshot_path 全量导入。"""

    workspace = make_workspace_tmp("cli")
    try:
        settings = Settings(project_root=workspace)
        write_jsonl(
            settings.snapshot_path,
            [
                {
                    "id": "2401.00001v2",
                    "title": "CLI Import Paper",
                    "abstract": "Imported from local snapshot.",
                    "authors": "Ada Lovelace",
                    "categories": "cs.AI cs.DB",
                    "versions": [],
                    "update_date": "2024-01-02",
                }
            ],
        )

        monkeypatch.setattr(cli, "DEFAULT_SETTINGS", settings)
        monkeypatch.setattr(sys, "argv", ["prog", "import-arxiv-snapshot"])

        cli.main()

        repo = PaperRepository(settings)
        paper = repo.get_paper("2401.00001")
        assert paper["title"] == "CLI Import Paper"
        assert paper["source"] == "arxiv_snapshot"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_sync_arxiv_incremental_cli_invokes_manual_update(monkeypatch, capsys):
    """CLI 中只保留手动增量更新命令，不再暴露旧同步命名。"""

    workspace = make_workspace_tmp("cli-incremental")
    try:
        settings = Settings(project_root=workspace)
        called = {}

        def fake_sync(received_settings):
            called["settings"] = received_settings
            return {"message": "手动增量更新完成"}

        monkeypatch.setattr(cli, "DEFAULT_SETTINGS", settings)
        monkeypatch.setattr(cli, "sync_arxiv_incremental", fake_sync)
        monkeypatch.setattr(sys, "argv", ["prog", "sync-arxiv-incremental"])

        cli.main()

        assert called["settings"] is settings
        assert "手动增量更新完成" in capsys.readouterr().out
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_precompute_clusters_cli_invokes_cache_builder(monkeypatch, capsys):
    """CLI 应提供 precompute-clusters，用于课设展示前离线生成所有 KMeans 缓存。"""

    workspace = make_workspace_tmp("cli-precompute-clusters")
    try:
        settings = Settings(project_root=workspace)
        called = {}

        def fake_precompute(received_settings, candidates, force, progress_callback):
            called["settings"] = received_settings
            called["candidates"] = list(candidates)
            called["force"] = force
            progress_callback({"stage": "kmeans", "completed": 0, "total": 2, "current_k": None})
            return {"cache_dir": str(received_settings.cluster_cache_dir)}

        monkeypatch.setattr(cli, "DEFAULT_SETTINGS", settings)
        monkeypatch.setattr(cli, "precompute_cluster_cache", fake_precompute)
        monkeypatch.setattr(sys, "argv", ["prog", "precompute-clusters", "--candidates", "4,6", "--force"])

        cli.main()

        assert called == {
            "settings": settings,
            "candidates": [4, 6],
            "force": True,
        }
        output = capsys.readouterr().out
        assert "[kmeans] 0/2" in output
        assert "聚类缓存预计算完成" in output
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
