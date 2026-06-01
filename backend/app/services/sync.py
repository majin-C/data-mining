from __future__ import annotations

import json
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from backend.app.config import Settings
from backend.app.datasets.arxiv_incremental import fetch_incremental_arxiv_records
from backend.app.db import PaperRepository
from backend.app.models.embedding import EmbeddingModel, build_missing_embeddings


LAST_INCREMENTAL_SUCCESS_KEY = "arxiv_incremental_last_success_utc"
INCREMENTAL_JOBS: dict[str, dict] = {}
INCREMENTAL_JOB_LOCK = threading.Lock()


def _utc_now() -> datetime:
    """返回当前 UTC 时间，方便测试通过参数注入固定时间。"""

    return datetime.now(timezone.utc)


def _parse_utc_datetime(value: str) -> datetime:
    """解析 app_state 或配置中的 UTC 时间字符串。"""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def _append_log(settings: Settings, message: str) -> None:
    """把手动增量更新过程写入本地日志文件，便于排错。"""

    settings.ensure_directories()
    timestamp = _utc_now().isoformat()
    with settings.sync_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def read_sync_status(settings: Settings, max_lines: int = 20) -> dict:
    """读取最近增量更新日志和上次成功更新时间。"""

    repo = PaperRepository(settings)
    lines: list[str] = []
    exists = settings.sync_log_path.exists()
    if exists:
        lines = settings.sync_log_path.read_text(encoding="utf-8").splitlines()[-max_lines:]
    return {
        "exists": exists,
        "lines": lines,
        "last_success_utc": repo.get_state(LAST_INCREMENTAL_SUCCESS_KEY),
    }


def _append_incremental_records(settings: Settings, records: list[dict]) -> None:
    """把手动抓取结果追加到增量 JSONL 文件。

    数据库用 arXiv ID 主键负责业务去重；JSONL 文件保留抓取痕迹，便于后续检查
    某次更新实际从 arXiv API 收到了哪些记录。
    """

    if not records:
        return
    settings.ensure_directories()
    with settings.incremental_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_window_start(settings: Settings, repo: PaperRepository) -> datetime:
    """读取手动增量更新的起点；首次运行使用当前全量数据集截止时间。"""

    saved = repo.get_state(LAST_INCREMENTAL_SUCCESS_KEY)
    if saved:
        return _parse_utc_datetime(saved)
    return _parse_utc_datetime(settings.initial_incremental_start_utc)


def _copy_job_state(job: dict) -> dict:
    """复制后台任务状态，避免 API 调用方误改全局任务字典。"""

    return json.loads(json.dumps(job, ensure_ascii=False))


def _save_job_state(job_id: str, **updates) -> dict:
    """线程安全地保存手动增量任务状态。"""

    with INCREMENTAL_JOB_LOCK:
        job = INCREMENTAL_JOBS.setdefault(
            job_id,
            {
                "job_id": job_id,
                "status": "running",
                "stage": "queued",
                "window_start_utc": None,
                "window_end_utc": None,
                "fetched": 0,
                "inserted": 0,
                "skipped": 0,
                "embedded": 0,
                "embedding_completed": 0,
                "embedding_total": 0,
                "result": None,
                "error": None,
            },
        )
        job.update(updates)
        return _copy_job_state(job)


def get_arxiv_incremental_job(job_id: str) -> dict:
    """读取指定手动增量任务的当前状态。"""

    with INCREMENTAL_JOB_LOCK:
        job = INCREMENTAL_JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return _copy_job_state(job)


def sync_arxiv_incremental(
    settings: Settings,
    model: EmbeddingModel | None = None,
    now: datetime | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """执行手动 arXiv 增量更新。

    只有抓取、入库和 embedding 追加全部成功后，才会推进上次成功更新时间。
    如果任一步骤失败，下次点击“更新数据”仍会从原窗口起点重新抓取，避免漏收论文。
    """

    repo = PaperRepository(settings)
    window_start = _load_window_start(settings, repo)
    window_end = (now or _utc_now()).astimezone(timezone.utc)

    def report(**updates) -> None:
        if progress_callback is not None:
            progress_callback(updates)

    try:
        report(
            stage="fetching",
            window_start_utc=window_start.isoformat(),
            window_end_utc=window_end.isoformat(),
        )
        records = fetch_incremental_arxiv_records(
            settings.target_categories,
            submitted_from=window_start,
            submitted_until=window_end,
            progress_callback=lambda event: report(stage="fetching", fetched=int(event.get("fetched", 0))),
        )
        report(stage="storing", fetched=len(records))

        candidate_ids = list(dict.fromkeys(str(record.get("id", "")) for record in records if record.get("id")))
        existing_ids = repo.existing_ids(candidate_ids)
        _append_incremental_records(settings, records)
        inserted = repo.upsert_papers(records, source="arxiv_incremental")
        skipped = len(records) - inserted
        report(stage="embedding", inserted=inserted, skipped=skipped, embedding_completed=0, embedding_total=0)

        def embedding_progress(event: dict) -> None:
            report(
                stage="embedding",
                embedding_completed=int(event.get("completed", 0)),
                embedding_total=int(event.get("total", 0)),
            )

        # 传入整个窗口的候选 ID，而不是只传新增 ID。这样如果窗口内已有论文曾因中断
        # 没有生成 embedding，本次手动更新会自动补齐缺失向量。
        embedded = build_missing_embeddings(
            settings,
            repo,
            model=model,
            paper_ids=candidate_ids,
            progress_callback=embedding_progress,
        )
        repo.set_state(LAST_INCREMENTAL_SUCCESS_KEY, window_end.isoformat())
        result = {
            "ok": True,
            "window_start_utc": window_start.isoformat(),
            "window_end_utc": window_end.isoformat(),
            "fetched": len(records),
            "inserted": inserted,
            "skipped": skipped,
            "embedded": embedded,
            "existing": len(existing_ids),
            "message": f"更新完成：获取 {len(records)} 篇，新增 {inserted} 篇，跳过 {skipped} 篇，生成向量 {embedded} 篇。",
        }
        report(stage="completed", **result)
        _append_log(settings, result["message"])
        return result
    except Exception as exc:
        error = f"更新失败：{exc}"
        _append_log(settings, error)
        _append_log(settings, traceback.format_exc())
        report(stage="failed", error=error)
        return {
            "ok": False,
            "window_start_utc": window_start.isoformat(),
            "window_end_utc": window_end.isoformat(),
            "fetched": 0,
            "inserted": 0,
            "skipped": 0,
            "embedded": 0,
            "existing": 0,
            "message": error,
        }


def start_arxiv_incremental_job(settings: Settings) -> dict:
    """启动后台手动增量更新任务，供前端轮询真实进度。"""

    job_id = uuid.uuid4().hex
    _save_job_state(job_id, status="running", stage="queued", error=None, result=None)

    def progress_callback(event: dict) -> None:
        """把同步服务的阶段性进度同步到全局任务状态。"""

        _save_job_state(job_id, status="running", **event)

    def run_job() -> None:
        """后台执行耗时的抓取和 embedding，避免前端请求长时间阻塞。"""

        result = sync_arxiv_incremental(settings, progress_callback=progress_callback)
        if result["ok"]:
            _save_job_state(
                job_id,
                status="completed",
                stage="completed",
                fetched=result["fetched"],
                inserted=result["inserted"],
                skipped=result["skipped"],
                embedded=result["embedded"],
                embedding_completed=result["embedded"],
                embedding_total=result["embedded"],
                result=result,
                error=None,
            )
        else:
            _save_job_state(job_id, status="failed", stage="failed", result=result, error=result["message"])

    thread = threading.Thread(target=run_job, name=f"arxiv-incremental-{job_id}", daemon=True)
    thread.start()
    return get_arxiv_incremental_job(job_id)
