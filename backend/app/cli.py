from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.config import DEFAULT_SETTINGS
from backend.app.datasets.local_arxiv import rebuild_from_snapshot
from backend.app.db import PaperRepository, init_db
from backend.app.models.embedding import build_missing_embeddings
from backend.app.services.clustering import K_EVALUATION_CANDIDATES, cluster_embeddings, precompute_cluster_cache
from backend.app.services.sync import sync_arxiv_incremental


def main() -> None:
    """项目命令行入口。"""

    parser = argparse.ArgumentParser(description="论文智能聚类与推荐系统命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="初始化 SQLite 数据库")

    import_parser = subparsers.add_parser("import-arxiv-snapshot", help="从本地 arXiv JSONL 快照全量重建数据库")
    import_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="本地 arXiv JSONL 快照路径；不传时使用 Settings.snapshot_path",
    )

    embedding_parser = subparsers.add_parser("build-embeddings", help="为未生成 embedding 的论文生成向量")
    embedding_parser.add_argument("--batch-size", type=int, default=16, help="模型批处理大小")

    cluster_parser = subparsers.add_parser("cluster", help="执行 KMeans 聚类和二维降维")
    cluster_parser.add_argument("--clusters", type=int, default=12, help="聚类数量 K")

    precompute_parser = subparsers.add_parser("precompute-clusters", help="预计算所有候选 K 的全量 KMeans 缓存")
    precompute_parser.add_argument("--force", action="store_true", help="覆盖已有聚类缓存并重新计算")
    precompute_parser.add_argument(
        "--candidates",
        type=str,
        default=",".join(str(k) for k in K_EVALUATION_CANDIDATES),
        help="候选 K 列表，默认 8,10,12,...,30",
    )

    subparsers.add_parser("sync-arxiv-incremental", help="按上次更新时间手动增量更新 arXiv 论文并生成向量")

    args = parser.parse_args()
    settings = DEFAULT_SETTINGS
    init_db(settings)
    repo = PaperRepository(settings)

    if args.command == "init-db":
        print(f"数据库已初始化：{settings.db_path}")
    elif args.command == "import-arxiv-snapshot":
        # 全量导入采用重建语义：清空旧数据库和旧向量文件后再导入本地快照。
        # embedding 生成仍由 build-embeddings 单独执行，避免导入阶段长时间占用 GPU。
        inserted = rebuild_from_snapshot(settings, source=args.source)
        print(f"本地 arXiv 快照导入完成，新增论文 {inserted} 篇。")
    elif args.command == "build-embeddings":
        count = build_missing_embeddings(settings, repo, batch_size=args.batch_size)
        print(f"embedding 生成完成，处理论文 {count} 篇。")
    elif args.command == "cluster":
        result = cluster_embeddings(settings, repo, n_clusters=args.clusters)
        print(f"聚类完成：{result}")
    elif args.command == "precompute-clusters":
        candidates = [int(value.strip()) for value in args.candidates.split(",") if value.strip()]

        def report_progress(event: dict) -> None:
            stage = event.get("stage", "kmeans")
            current_category = event.get("current_category")
            current_k = event.get("current_k")
            completed = event.get("completed")
            total = event.get("total")
            # 分组聚类预计算会先处理主分类，再在分类内部逐个 K 训练；
            # 命令行输出分类名可以让长时间运行时更容易判断当前进度。
            prefix = f"[{stage}]"
            if current_category:
                prefix += f" {current_category}"
            if current_k is None:
                print(f"{prefix} {completed}/{total}")
            else:
                print(f"{prefix} K={current_k} {completed}/{total}")

        result = precompute_cluster_cache(
            settings,
            candidates=candidates,
            force=args.force,
            progress_callback=report_progress,
        )
        print(f"聚类缓存预计算完成：{result['cache_dir']}")
    elif args.command == "sync-arxiv-incremental":
        result = sync_arxiv_incremental(settings)
        print(result["message"])


if __name__ == "__main__":
    main()
