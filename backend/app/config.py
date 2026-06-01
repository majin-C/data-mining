from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet


@dataclass(slots=True)
class Settings:
    """集中管理项目运行配置。

    课程设计项目会同时被命令行脚本、FastAPI 服务和测试使用。把路径、模型名、
    目标分类和镜像配置集中在一个对象里，可以避免多个模块各自拼路径导致不一致。
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    target_categories: FrozenSet[str] = frozenset(
        {"cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.DB", "cs.SE", "cs.DS"}
    )
    specter_base_model: str = "allenai/specter2_base"
    specter_adapter_model: str = "allenai/specter2"
    hf_endpoint: str = "https://hf-mirror.com"
    initial_incremental_start_utc: str = "2026-05-30T23:59:00+00:00"

    def __post_init__(self) -> None:
        # dataclass 可能收到字符串路径，这里统一转成 Path，后续模块就不用重复判断类型。
        self.project_root = Path(self.project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cluster_cache_dir(self) -> Path:
        return self.processed_data_dir / "cluster_cache"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def paper_ids_path(self) -> Path:
        return self.data_dir / "paper_ids.json"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "model_cache"

    @property
    def snapshot_path(self) -> Path:
        """返回本地全量 arXiv 快照路径。

        该文件是当前系统的初始数据源，包含 8 个目标 CS 分区在 2026-05-30
        23:59 前去重后的论文记录。
        """

        return self.raw_data_dir / "arxiv_cs_ai_8cats_until_20260530.json"

    @property
    def incremental_path(self) -> Path:
        """返回手动增量更新 JSONL 文件路径。

        前端手动点击“更新数据”抓取的新论文会先追加到这个文件，再写入 SQLite。
        保留原始增量记录可以在课程答辩或排错时追溯每次更新收录了哪些论文。
        """

        return self.raw_data_dir / "arxiv_incremental.jsonl"

    @property
    def sync_log_path(self) -> Path:
        return self.logs_dir / "arxiv_incremental.log"

    def ensure_directories(self) -> None:
        """创建运行所需目录。

        目录创建是幂等操作，命令行、API 和测试可以安全地重复调用。
        """

        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.logs_dir,
            self.model_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def apply_huggingface_mirror(self) -> None:
        """设置 Hugging Face 镜像环境变量。

        transformers/adapters 会读取 `HF_ENDPOINT`，因此在加载模型前统一设置。
        如果用户已经显式配置了其他 endpoint，这里仍按课程设计要求覆盖到镜像站。
        """

        # 当前机器上可能存在指向 127.0.0.1:9 的无效代理变量。
        # huggingface_hub 会自动读取这些变量；如果不清理，即使 HF_ENDPOINT 指向镜像站，
        # Python 请求也会先走无效代理并失败。
        for proxy_key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "GIT_HTTP_PROXY",
            "GIT_HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "git_http_proxy",
            "git_https_proxy",
        ):
            os.environ.pop(proxy_key, None)

        os.environ["HF_ENDPOINT"] = self.hf_endpoint
        os.environ["HF_HOME"] = str(self.model_cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.model_cache_dir / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.model_cache_dir / "hub")


DEFAULT_SETTINGS = Settings(project_root=Path(__file__).resolve().parents[2])
