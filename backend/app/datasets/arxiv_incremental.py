from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.app.datasets.local_arxiv import clean_text


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)?")


@dataclass(slots=True)
class ArxivEntry:
    """arXiv API Atom 条目的中间表示。"""

    raw_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    doi: str = ""
    comment: str = ""


def _parse_datetime(value: str) -> datetime:
    """解析 arXiv Atom 时间字符串，并统一保留 UTC 时区信息。"""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def _format_arxiv_submitted_date(value: datetime) -> str:
    """把 UTC 时间转换为 arXiv API submittedDate 范围查询需要的格式。"""

    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def _text(element: ET.Element | None) -> str:
    """安全读取 XML 文本节点。"""

    return clean_text(element.text if element is not None else "")


def parse_arxiv_feed(
    xml_text: str,
    submitted_from: datetime | None = None,
    submitted_until: datetime | None = None,
) -> list[ArxivEntry]:
    """解析 arXiv API 返回的 Atom XML，并按首次提交时间做窗口过滤。

    arXiv API 的 submittedDate 查询已经会限制大范围；这里再按 published 时间
    做一次本地过滤，避免边界分钟或 API 返回冗余结果时把窗口外论文写入数据库。
    """

    root = ET.fromstring(xml_text)
    entries: list[ArxivEntry] = []
    lower_bound = submitted_from.astimezone(timezone.utc) if submitted_from is not None else None
    upper_bound = submitted_until.astimezone(timezone.utc) if submitted_until is not None else None
    for entry in root.findall("atom:entry", ATOM_NS):
        published = _parse_datetime(_text(entry.find("atom:published", ATOM_NS)))
        if lower_bound is not None and published < lower_bound:
            continue
        if upper_bound is not None and published > upper_bound:
            continue
        entries.append(
            ArxivEntry(
                raw_id=_text(entry.find("atom:id", ATOM_NS)),
                title=_text(entry.find("atom:title", ATOM_NS)),
                summary=_text(entry.find("atom:summary", ATOM_NS)),
                authors=[
                    _text(author.find("atom:name", ATOM_NS))
                    for author in entry.findall("atom:author", ATOM_NS)
                ],
                categories=[
                    category.attrib.get("term", "")
                    for category in entry.findall("atom:category", ATOM_NS)
                    if category.attrib.get("term")
                ],
                published=published,
                updated=_parse_datetime(_text(entry.find("atom:updated", ATOM_NS))),
                doi=_text(entry.find("arxiv:doi", ATOM_NS)),
                comment=_text(entry.find("arxiv:comment", ATOM_NS)),
            )
        )
    return entries


def _extract_arxiv_id(raw_id: str) -> tuple[str, str]:
    """从 arXiv URL 中提取无版本 ID 和版本号。"""

    match = ARXIV_ID_RE.search(raw_id)
    if not match:
        return raw_id.rsplit("/", maxsplit=1)[-1], "v1"
    return match.group("id"), match.group("version") or "v1"


def map_arxiv_entry_to_snapshot_record(entry: ArxivEntry) -> dict:
    """把 arXiv API 条目映射成与本地快照一致的字段。"""

    arxiv_id, version = _extract_arxiv_id(entry.raw_id)
    # Windows 的 strftime 不支持 "%-d"，这里手动拼接 day，保证脚本跨平台可运行。
    created = (
        f"{entry.published.strftime('%a')}, {entry.published.day} "
        f"{entry.published.strftime('%b %Y %H:%M:%S GMT')}"
    )
    return {
        "id": arxiv_id,
        "title": entry.title,
        "abstract": entry.summary,
        "authors": ", ".join(entry.authors),
        "categories": " ".join(entry.categories),
        "versions": [
            {
                "version": version,
                "created": created,
            }
        ],
        "update_date": entry.updated.date().isoformat(),
        "doi": entry.doi,
        "journal_ref": "",
        "comments": entry.comment,
        "source": "arxiv_incremental",
    }


def build_incremental_query(categories: set[str] | frozenset[str], submitted_from: datetime, submitted_until: datetime) -> str:
    """构造 arXiv API 查询语句，限定 8 个目标分区和首次提交时间窗口。"""

    category_query = " OR ".join(f"cat:{category}" for category in sorted(categories))
    date_range = (
        f"submittedDate:[{_format_arxiv_submitted_date(submitted_from)} "
        f"TO {_format_arxiv_submitted_date(submitted_until)}]"
    )
    return f"({category_query}) AND {date_range}"


def fetch_incremental_arxiv_records(
    categories: set[str] | frozenset[str],
    submitted_from: datetime,
    submitted_until: datetime,
    page_size: int = 100,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    """分页读取指定时间窗口内的 arXiv 新提交论文元数据。

    arXiv API 通过 `start` 和 `max_results` 分页；只要当前页返回数量等于 page_size，
    就继续请求下一页，直到返回较短页面为止，确保窗口内记录不会被单页限制截断。
    """

    records: list[dict] = []
    start = 0
    while True:
        params = {
            "search_query": build_incremental_query(categories, submitted_from, submitted_until),
            "start": str(start),
            "max_results": str(page_size),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_text = response.read().decode("utf-8")
        page_entries = parse_arxiv_feed(xml_text, submitted_from=submitted_from, submitted_until=submitted_until)
        records.extend(map_arxiv_entry_to_snapshot_record(entry) for entry in page_entries)
        if progress_callback is not None:
            progress_callback({"fetched": len(records), "page_size": len(page_entries), "start": start})
        if len(page_entries) < page_size:
            break
        start += page_size
    return records
