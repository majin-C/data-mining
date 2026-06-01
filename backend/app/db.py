from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

from backend.app.config import Settings


FTS_TITLE_WEIGHT = 8.0
FTS_ABSTRACT_WEIGHT = 1.0


def _utc_now() -> str:
    """返回 ISO 格式 UTC 时间，保证数据库时间字段跨时区可比较。"""

    return datetime.now(timezone.utc).isoformat()


def parse_arxiv_created_date(versions: Iterable[dict[str, Any]] | str | None) -> str:
    """从 arXiv versions 字段中提取最早创建日期。

    本地快照和手动增量更新都把首次提交时间放在 `versions[].created` 中，常见格式是
    `Mon, 1 Jan 2024 00:00:00 GMT`。前端时间筛选只需要日期粒度，因此这里统一转成
    `YYYY-MM-DD`；无法解析时返回空字符串，让筛选逻辑自然排除该记录。
    """

    if not versions:
        return ""
    if isinstance(versions, str):
        try:
            versions = json.loads(versions)
        except json.JSONDecodeError:
            return ""

    parsed_dates: list[datetime] = []
    for item in versions:
        if not isinstance(item, dict):
            continue
        created = str(item.get("created") or "").strip()
        if not created:
            continue
        try:
            parsed_dates.append(parsedate_to_datetime(created))
            continue
        except (TypeError, ValueError):
            pass
        try:
            parsed_dates.append(datetime.fromisoformat(created.replace("Z", "+00:00")))
        except ValueError:
            continue
    if not parsed_dates:
        return ""
    return min(parsed_dates).date().isoformat()


def connect(settings: Settings) -> sqlite3.Connection:
    """创建 SQLite 连接，并启用字典式行读取。

    SQLite 足够支撑课程设计的数千篇论文数据，部署成本也低于独立数据库服务。
    """

    settings.ensure_directories()
    connection = sqlite3.connect(settings.db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(settings: Settings) -> None:
    """初始化数据库 schema。

    论文表同时保存元数据、embedding 是否生成、聚类标签和二维坐标。这样前端
    读取搜索结果或聚类图时，不需要再额外拼接多个文件。
    """

    with connect(settings) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL DEFAULT '',
                categories TEXT NOT NULL DEFAULT '',
                versions TEXT NOT NULL DEFAULT '[]',
                created_date TEXT NOT NULL DEFAULT '',
                update_date TEXT NOT NULL DEFAULT '',
                doi TEXT NOT NULL DEFAULT '',
                journal_ref TEXT NOT NULL DEFAULT '',
                comments TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'arxiv_snapshot',
                embedding_ready INTEGER NOT NULL DEFAULT 0,
                cluster_id INTEGER,
                x REAL,
                y REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_created_date_column(connection)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_papers_categories
            ON papers(categories)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_papers_cluster
            ON papers(cluster_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_papers_update_date
            ON papers(update_date)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_papers_created_date
            ON papers(created_date)
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
            USING fts5(
                paper_id UNINDEXED,
                title,
                abstract,
                tokenize = 'unicode61'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _backfill_created_dates(connection)
        _rebuild_fts_if_needed(connection)


def _ensure_created_date_column(connection: sqlite3.Connection) -> None:
    """为旧数据库补齐 created_date 列。

    早期版本只保存原始 `versions` JSON，没有单独的创建日期列；新增时间筛选后，
    将创建日期落成普通 TEXT 列可以直接用 SQLite 做范围查询，避免每次搜索都扫描并解析
    61 万条 JSON。
    """

    columns = {row["name"] for row in connection.execute("PRAGMA table_info(papers)").fetchall()}
    if "created_date" not in columns:
        connection.execute("ALTER TABLE papers ADD COLUMN created_date TEXT NOT NULL DEFAULT ''")


def _backfill_created_dates(connection: sqlite3.Connection) -> None:
    """把旧数据的 versions.created 回填到 created_date。

    该步骤是幂等的：只有 created_date 为空的记录会被处理。用户已有的 615819 篇论文库
    在首次启动新后端时会完成一次回填，之后不会重复解析已写入日期的记录。
    """

    rows = connection.execute(
        "SELECT id, versions FROM papers WHERE created_date = '' OR created_date IS NULL"
    ).fetchall()
    updates = []
    for row in rows:
        created_date = parse_arxiv_created_date(row["versions"])
        if created_date:
            updates.append((created_date, row["id"]))
    if updates:
        connection.executemany("UPDATE papers SET created_date = ? WHERE id = ?", updates)


def _rebuild_fts_if_needed(connection: sqlite3.Connection) -> None:
    """旧数据库首次升级到 FTS5 搜索时，自动补建标题/摘要倒排索引。

    FTS 表与 papers 表分开维护，便于用 SQLite 内置 `bm25()` 做相关度排序。
    如果两边数量不一致，说明可能是旧库还没建索引或历史写入中断，直接重建一遍
    能保证后续 `/api/papers?query=...` 搜索结果完整。
    """

    paper_count = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    fts_count = connection.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0]
    if paper_count == fts_count:
        return
    connection.execute("DELETE FROM papers_fts")
    connection.execute(
        """
        INSERT INTO papers_fts(paper_id, title, abstract)
        SELECT id, title, abstract
        FROM papers
        ORDER BY id
        """
    )


def _build_fts_query(query: str | None) -> str:
    """把用户输入转换成安全的 FTS5 MATCH 查询。

    FTS5 的 MATCH 语法会把冒号、括号、加号等字符解释成操作符；前端搜索框是普通
    关键词输入，因此这里只保留 unicode 单词字符，并用空格连接成隐式 AND 查询。
    如果用户只输入标点符号，会返回空字符串，调用方按“无关键词”处理。
    """

    if not query:
        return ""
    tokens = re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
    return " ".join(tokens)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """把 SQLite Row 转成 API 友好的字典，并反序列化 versions 字段。"""

    data = dict(row)
    try:
        data["versions"] = json.loads(data.get("versions") or "[]")
    except json.JSONDecodeError:
        # 如果历史数据里 versions 字段损坏，返回空列表可以保证页面仍能展示其他信息。
        data["versions"] = []
    return data


class PaperRepository:
    """论文数据访问层。

    上层服务只通过这个类读写论文，避免 SQL 分散在 API、同步和算法代码中。
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def upsert_papers(self, papers: Iterable[dict[str, Any]], source: str = "arxiv_snapshot") -> int:
        """按 arXiv ID 去重插入论文，已存在的论文直接跳过。

        返回实际新增数量，便于手动增量更新日志展示“新增/跳过”的差异。
        """

        inserted = 0
        now = _utc_now()
        with connect(self.settings) as connection:
            for paper in papers:
                paper_id = paper["id"]
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO papers (
                        id, title, abstract, authors, categories, versions, created_date, update_date,
                        doi, journal_ref, comments, source, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        title,
                        abstract,
                        paper.get("authors", ""),
                        paper.get("categories", ""),
                        json.dumps(paper.get("versions", []), ensure_ascii=False),
                        parse_arxiv_created_date(paper.get("versions")),
                        paper.get("update_date", ""),
                        paper.get("doi", ""),
                        paper.get("journal_ref", paper.get("journal-ref", "")),
                        paper.get("comments", ""),
                        paper.get("source", source),
                        now,
                        now,
                    ),
                )
                if cursor.rowcount:
                    # FTS5 虚拟表不负责业务去重，只有 papers 主表真正插入成功时才同步写入。
                    # 这样手动增量更新遇到已存在论文时，不会在倒排索引里产生重复文档。
                    connection.execute(
                        """
                        INSERT INTO papers_fts(paper_id, title, abstract)
                        VALUES (?, ?, ?)
                        """,
                        (paper_id, title, abstract),
                    )
                    inserted += 1
        return inserted

    def mark_embedding_ready(self, paper_ids: Iterable[str]) -> None:
        """标记指定论文已经生成 embedding。"""

        ids = list(paper_ids)
        if not ids:
            return
        now = _utc_now()
        with connect(self.settings) as connection:
            connection.executemany(
                "UPDATE papers SET embedding_ready = 1, updated_at = ? WHERE id = ?",
                [(now, paper_id) for paper_id in ids],
            )

    def update_cluster_results(self, results: dict[str, dict[str, float | int]]) -> None:
        """把聚类编号和二维坐标写回数据库。"""

        if not results:
            return
        now = _utc_now()
        with connect(self.settings) as connection:
            connection.executemany(
                """
                UPDATE papers
                SET cluster_id = ?, x = ?, y = ?, updated_at = ?
                WHERE id = ?
                """,
                [
                    (
                        int(value["cluster_id"]),
                        float(value["x"]),
                        float(value["y"]),
                        now,
                        paper_id,
                    )
                    for paper_id, value in results.items()
                ],
            )

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        """按 ID 读取单篇论文；不存在时抛出 KeyError 交给 API 转 404。"""

        with connect(self.settings) as connection:
            row = connection.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            raise KeyError(paper_id)
        return _row_to_dict(row)

    def get_papers_by_ids(self, paper_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """按 ID 批量读取论文元数据。

        聚类缓存散点图只需要 5000 个采样点的标题和分类；批量查询比逐篇调用
        `get_paper` 少很多 SQLite 往返，也避免把 61 万篇论文一次性读入 API 响应路径。
        """

        ids = list(dict.fromkeys(str(paper_id) for paper_id in paper_ids))
        if not ids:
            return {}
        rows = []
        with connect(self.settings) as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    connection.execute(
                        f"SELECT id, title, categories FROM papers WHERE id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
        return {str(row["id"]): _row_to_dict(row) for row in rows}

    def basic_stats(self) -> dict[str, int]:
        """返回不依赖聚类字段的基础统计。

        当前聚类展示改为读取本地 KMeans 缓存，`papers.cluster_id/x/y` 只作为历史字段保留；
        因此统计接口需要先拿论文数和向量数，再由缓存层补充聚类统计。
        """

        with connect(self.settings) as connection:
            paper_count = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            embedding_count = connection.execute(
                "SELECT COUNT(*) FROM papers WHERE embedding_ready = 1"
            ).fetchone()[0]
        return {"paper_count": int(paper_count), "embedding_count": int(embedding_count)}

    def search_papers(
        self,
        query: str | None = None,
        category: str | None = None,
        cluster_id: int | None = None,
        time_field: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """按关键词、分类、聚类编号和时间范围分页查询论文。

        `time_field` 只允许 `update_date` 或 `created_date`，避免用户传入任意 SQL 字段名。
        日期统一采用 `YYYY-MM-DD` 文本比较，这与数据库存储格式一致，范围查询可直接使用索引语义。
        """

        fts_query = _build_fts_query(query)
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("papers.categories LIKE ?")
            params.append(f"%{category}%")
        if cluster_id is not None:
            clauses.append("papers.cluster_id = ?")
            params.append(cluster_id)
        if time_field in {"update_date", "created_date"}:
            if date_from:
                clauses.append(f"papers.{time_field} >= ?")
                params.append(date_from)
            if date_to:
                clauses.append(f"papers.{time_field} <= ?")
                params.append(date_to)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = max(page - 1, 0) * page_size

        with connect(self.settings) as connection:
            if fts_query:
                # FTS5 的 bm25() 返回值越小相关度越高；标题权重大于摘要，让标题命中排在前面。
                # 其他筛选条件仍在 papers 主表执行，保证分类、聚类和时间过滤语义不变。
                fts_where = f"WHERE papers_fts MATCH ? {'AND ' + ' AND '.join(clauses) if clauses else ''}"
                fts_params = [fts_query, *params]
                total = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM papers
                    JOIN papers_fts ON papers.id = papers_fts.paper_id
                    {fts_where}
                    """,
                    fts_params,
                ).fetchone()[0]
                rows = connection.execute(
                    f"""
                    SELECT papers.*
                    FROM papers
                    JOIN papers_fts ON papers.id = papers_fts.paper_id
                    {fts_where}
                    ORDER BY bm25(papers_fts, {FTS_TITLE_WEIGHT}, {FTS_ABSTRACT_WEIGHT}) ASC,
                             papers.update_date DESC,
                             papers.id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*fts_params, page_size, offset],
                ).fetchall()
                return [_row_to_dict(row) for row in rows], int(total)

            total = connection.execute(f"SELECT COUNT(*) FROM papers {where_sql}", params).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM papers
                {where_sql}
                ORDER BY update_date DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()
        return [_row_to_dict(row) for row in rows], int(total)

    def list_all_papers(self, only_embedding_ready: bool = False) -> list[dict[str, Any]]:
        """读取全部论文，算法模块用它与 embedding 文件对齐。"""

        where = "WHERE embedding_ready = 1" if only_embedding_ready else ""
        with connect(self.settings) as connection:
            rows = connection.execute(f"SELECT * FROM papers {where} ORDER BY id").fetchall()
        return [_row_to_dict(row) for row in rows]

    def existing_ids(self, paper_ids: Iterable[str]) -> set[str]:
        """读取数据库中已经存在的论文 ID。

        手动增量更新会先判断哪些 arXiv ID 已经入库，再只为本次窗口内缺失
        embedding 的论文生成向量，避免重复写入已存在向量。
        """

        ids = list(dict.fromkeys(paper_ids))
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        with connect(self.settings) as connection:
            rows = connection.execute(
                f"SELECT id FROM papers WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return {str(row["id"]) for row in rows}

    def list_missing_embedding_papers(self, paper_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """读取尚未生成 embedding 的论文。

        `paper_ids` 不为空时只检查指定论文，供手动增量更新精确处理本次窗口数据。
        """

        with connect(self.settings) as connection:
            if paper_ids is not None:
                ids = list(dict.fromkeys(paper_ids))
                if not ids:
                    return []
                placeholders = ",".join("?" for _ in ids)
                rows = connection.execute(
                    f"""
                    SELECT * FROM papers
                    WHERE embedding_ready = 0 AND id IN ({placeholders})
                    ORDER BY id
                    """,
                    ids,
                ).fetchall()
                return [_row_to_dict(row) for row in rows]
            rows = connection.execute(
                "SELECT * FROM papers WHERE embedding_ready = 0 ORDER BY id"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def cluster_points(self) -> list[dict[str, Any]]:
        """读取已有二维坐标的论文，用于前端聚类散点图。"""

        with connect(self.settings) as connection:
            rows = connection.execute(
                """
                SELECT id, title, categories, cluster_id, x, y
                FROM papers
                WHERE x IS NOT NULL AND y IS NOT NULL AND cluster_id IS NOT NULL
                ORDER BY cluster_id, id
                """
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        """返回系统统计信息，供首页仪表盘展示。"""

        with connect(self.settings) as connection:
            paper_count = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            embedding_count = connection.execute(
                "SELECT COUNT(*) FROM papers WHERE embedding_ready = 1"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT cluster_id, COUNT(*) AS count
                FROM papers
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
                ORDER BY cluster_id
                """
            ).fetchall()
            silhouette_row = connection.execute(
                "SELECT value FROM app_state WHERE key = 'silhouette_score'"
            ).fetchone()
        clusters = [{"cluster_id": row["cluster_id"], "count": row["count"]} for row in rows]
        return {
            "paper_count": int(paper_count),
            "embedding_count": int(embedding_count),
            "cluster_count": len(clusters),
            "clusters": clusters,
            "silhouette_score": float(silhouette_row["value"]) if silhouette_row else None,
        }

    def category_counts(self, target_categories: Iterable[str]) -> list[dict[str, Any]]:
        """统计目标分类在论文表中的命中数量。

        数据库中的 `categories` 是空格分隔字符串，同一篇论文可能同时属于多个分类。
        因此前端分类筛选需要的是“每个目标分类命中了多少篇论文”，而不是互斥分组。
        """

        items: list[dict[str, Any]] = []
        with connect(self.settings) as connection:
            for category in sorted(target_categories):
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM papers
                    WHERE (' ' || categories || ' ') LIKE ?
                    """,
                    (f"% {category} %",),
                ).fetchone()
                items.append({"id": category, "count": int(row["count"])})
        return items

    def set_state(self, key: str, value: str) -> None:
        """保存简单状态，例如轮廓系数。"""

        now = _utc_now()
        with connect(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO app_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def get_state(self, key: str) -> str | None:
        """读取简单状态值；状态不存在时返回 None。

        手动增量更新会用它保存上一次成功更新到的 UTC 时间。这个值只有在抓取、
        入库和 embedding 追加全部成功后才会推进，避免失败后下一次更新漏抓论文。
        """

        with connect(self.settings) as connection:
            row = connection.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None
