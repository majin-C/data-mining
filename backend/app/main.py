from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.config import DEFAULT_SETTINGS, Settings
from backend.app.db import PaperRepository, init_db
from backend.app.models.embedding import HashEmbeddingModel, SpecterEmbeddingModel
from backend.app.services.clustering import (
    activate_cluster_cache,
    active_cluster_points,
    active_cluster_summary,
    evaluate_k_candidates,
    get_k_evaluation_job,
    start_k_evaluation_job,
)
from backend.app.services.recommendation import top_k_similar
from backend.app.services.sync import get_arxiv_incremental_job, read_sync_status, start_arxiv_incremental_job
from backend.app.services.vector_store import load_embeddings


class QueryRecommendationRequest(BaseModel):
    """查询文本推荐请求体。"""

    query: str
    top_k: int = 10
    time_field: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class ApplyClusterRequest(BaseModel):
    """前端提交的分组 KMeans 应用请求体。

    selected 的键是 arXiv 主分类，值是该分类内部要使用的子聚类 K。
    后端只切换已经预计算好的本地缓存，不在接口请求里现场训练 KMeans。
    """

    selected: dict[str, int] = Field(default_factory=dict)


class EvaluateClusterRequest(BaseModel):
    """K 值评估请求体。

    `force=False` 表示优先读取本地缓存；`force=True` 表示用户主动点击“重新评估”，
    后端会忽略已有缓存并重新计算 8-30 候选 K 的轮廓系数。
    """

    force: bool = False


def _paper_summary(paper: dict, score: float | None = None) -> dict:
    """把数据库论文记录转换为前端列表项。"""

    item = {
        "id": paper["id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper["authors"],
        "categories": paper["categories"],
        "update_date": paper.get("update_date"),
        "created_date": paper.get("created_date"),
        "cluster_id": paper.get("cluster_id"),
        "x": paper.get("x"),
        "y": paper.get("y"),
    }
    if score is not None:
        item["score"] = score
    return item


def _has_time_filter(time_field: str | None, date_from: str | None, date_to: str | None) -> bool:
    """判断请求是否启用了时间筛选。"""

    return time_field in {"update_date", "created_date"} and bool(date_from or date_to)


def _paper_matches_time_filter(
    paper: dict,
    time_field: str | None,
    date_from: str | None,
    date_to: str | None,
) -> bool:
    """复用论文列表同一套时间筛选规则过滤推荐结果。

    推荐算法先按向量相似度排序，再在排序结果中保留时间范围内的论文；
    这样既保留“相似度优先”的推荐语义，又能保证前端推荐列表不会出现超出时间条件的论文。
    """

    if not _has_time_filter(time_field, date_from, date_to):
        return True
    value = str(paper.get(time_field or "") or "")
    if not value:
        return False
    if date_from and value < date_from:
        return False
    if date_to and value > date_to:
        return False
    return True


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    当前系统已取消自动定时同步，数据更新统一由前端“更新数据”按钮触发后台手动增量任务。
    """

    settings = settings or DEFAULT_SETTINGS
    init_db(settings)
    app = FastAPI(title="论文智能聚类与推荐系统", version="1.0.0")
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/papers")
    def list_papers(
        query: str | None = None,
        category: str | None = None,
        cluster_id: int | None = None,
        time_field: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
    ) -> dict:
        """分页查询论文列表。"""

        repo = PaperRepository(settings)
        items, total = repo.search_papers(
            query,
            category,
            cluster_id,
            time_field=time_field,
            date_from=date_from,
            date_to=date_to,
            page=page,
            page_size=page_size,
        )
        return {"items": [_paper_summary(item) for item in items], "total": total, "page": page, "page_size": page_size}

    @app.get("/api/papers/{paper_id}")
    def get_paper(paper_id: str) -> dict:
        """读取论文详情。"""

        repo = PaperRepository(settings)
        try:
            return repo.get_paper(paper_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="论文不存在")

    @app.get("/api/papers/{paper_id}/recommendations")
    def recommend_by_paper(
        paper_id: str,
        top_k: int = Query(10, ge=1, le=50),
        time_field: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict:
        """基于选中论文的 embedding 推荐相似论文。"""

        repo = PaperRepository(settings)
        paper_ids, embeddings = load_embeddings(settings)
        if len(paper_ids) == 0:
            raise HTTPException(status_code=400, detail="尚未生成 embedding，请先运行 build-embeddings。")
        try:
            # 时间筛选可能排除大量高相似论文，因此启用筛选时先取完整相似度排序，
            # 再按时间范围截取前 top_k 条，保证推荐结果数量和筛选条件都尽量满足。
            candidate_limit = len(paper_ids) if _has_time_filter(time_field, date_from, date_to) else top_k
            scored = top_k_similar(paper_id, paper_ids, embeddings, top_k=candidate_limit)
        except KeyError:
            raise HTTPException(status_code=404, detail="论文不存在或尚未生成 embedding")
        items = []
        for item in scored:
            paper = repo.get_paper(str(item["id"]))
            if not _paper_matches_time_filter(paper, time_field, date_from, date_to):
                continue
            items.append(_paper_summary(paper, score=float(item["score"])))
            if len(items) >= top_k:
                break
        return {"items": items}

    @app.post("/api/recommendations/query")
    def recommend_by_query(payload: QueryRecommendationRequest) -> dict:
        """基于用户输入的研究兴趣文本推荐论文。"""

        repo = PaperRepository(settings)
        paper_ids, embeddings = load_embeddings(settings)
        if len(paper_ids) == 0:
            raise HTTPException(status_code=400, detail="尚未生成 embedding，请先运行 build-embeddings。")
        try:
            encoder = SpecterEmbeddingModel(settings)
            query_vector = encoder.encode([payload.query])[0]
        except Exception:
            # 如果模型依赖尚未安装，使用哈希 embedding 兜底，让演示系统仍能返回结果。
            query_vector = HashEmbeddingModel(dimension=embeddings.shape[1]).encode([payload.query])[0]
        candidate_limit = (
            len(paper_ids)
            if _has_time_filter(payload.time_field, payload.date_from, payload.date_to)
            else payload.top_k
        )
        scored = top_k_similar(None, paper_ids, embeddings, top_k=candidate_limit, query_vector=query_vector)
        items = []
        for item in scored:
            paper = repo.get_paper(str(item["id"]))
            if not _paper_matches_time_filter(paper, payload.time_field, payload.date_from, payload.date_to):
                continue
            items.append(_paper_summary(paper, score=float(item["score"])))
            if len(items) >= payload.top_k:
                break
        return {"items": items}

    @app.get("/api/clusters/plot")
    def cluster_plot() -> dict:
        """返回 8 个 arXiv 主分类各自的二维聚类散点图数据。"""

        repo = PaperRepository(settings)
        try:
            return {"groups": active_cluster_points(settings, repo)}
        except ValueError:
            return {"groups": []}

    @app.post("/api/clusters/evaluate-k")
    def evaluate_cluster_k(payload: EvaluateClusterRequest | None = None) -> dict:
        """计算 K 值评估折线图数据。

        普通评估优先复用本地 JSON 缓存，主动重新评估才覆盖缓存；
        该接口只做固定 20000 条样本的轮廓系数评估，不写数据库；
        前端用户点击某个 K 并确认应用后，才会调用 `/api/clusters/apply-k` 更新聚类结果。
        """

        try:
            return evaluate_k_candidates(settings, force=payload.force if payload else False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/clusters/evaluate-k/start")
    def start_evaluate_cluster_k(payload: EvaluateClusterRequest | None = None) -> dict:
        """启动带进度的 K 值评估任务。

        该接口会立即返回任务状态；前端随后轮询 `/api/clusters/evaluate-k/status/{job_id}`，
        以便展示真实的候选 K 评估进度，而不是只显示静态加载动画。
        """

        return start_k_evaluation_job(settings, force=payload.force if payload else False)

    @app.get("/api/clusters/evaluate-k/status/{job_id}")
    def evaluate_cluster_k_status(job_id: str) -> dict:
        """读取 K 值评估任务进度。"""

        try:
            return get_k_evaluation_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="K 值评估任务不存在") from exc

    @app.post("/api/clusters/apply-k")
    def apply_cluster_k(payload: ApplyClusterRequest) -> dict:
        """按用户选择的 8 个分类 K 值切换到对应的本地预计算缓存。"""

        repo = PaperRepository(settings)
        try:
            return activate_cluster_cache(settings, repo, payload.selected)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/stats")
    def stats() -> dict:
        """返回系统统计指标。"""

        repo = PaperRepository(settings)
        base = repo.basic_stats()
        try:
            cluster_summary = active_cluster_summary(settings, repo)
        except ValueError:
            cluster_summary = None
        if cluster_summary is None:
            return {
                **base,
                "cluster_count": 0,
                "clusters": [],
                "silhouette_score": None,
            }
        return {**base, **cluster_summary}

    @app.get("/api/categories")
    def categories() -> dict:
        """返回前端分类筛选项。

        分类来源于后端配置，前端不再硬编码具体分区；后续调整数据集分类范围时，
        只需要修改后端配置即可同步到页面。
        """

        repo = PaperRepository(settings)
        return {"items": repo.category_counts(settings.target_categories)}

    @app.post("/api/sync/arxiv-incremental/start")
    def start_incremental_update() -> dict:
        """启动手动 arXiv 增量更新后台任务。"""

        return start_arxiv_incremental_job(settings)

    @app.get("/api/sync/arxiv-incremental/status/{job_id}")
    def incremental_update_status(job_id: str) -> dict:
        """读取手动 arXiv 增量更新任务进度。"""

        try:
            return get_arxiv_incremental_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="增量更新任务不存在") from exc

    @app.get("/api/sync/status")
    def sync_status() -> dict:
        """读取最近手动增量更新日志。"""

        return read_sync_status(settings)

    return app


app = create_app()
