"""CLI 入口。

用法：
    python -m src.main "diffusion models for image super-resolution"
    python -m src.main "MicroLED 缺陷检测" --sources arxiv,openalex --top-fulltext 10
    python -m src.main "xxx" --provider stub --dry-run     # 离线试跑，不花 token
    python -m src.main "xxx" --human                       # 定稿前人工审核
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # 第三方库降噪
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lit-review",
        description="lit-review-agent：给定研究主题，自动检索、聚类并撰写带引用的文献综述。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("topic", nargs="?", help="研究主题（英文效果更好，中文也支持）")
    p.add_argument("-c", "--constraints", default="", help="额外约束，如「只看 2020 年后的工作」")

    g = p.add_argument_group("检索")
    g.add_argument("--sources", help="启用的检索源，逗号分隔：arxiv,openalex,semantic_scholar,pubmed,crossref")
    g.add_argument("-n", "--target", type=int, help="文献池目标规模（默认 40）")
    g.add_argument("--per-query", type=int, help="单条检索式单源最大返回条数（默认 25）")
    g.add_argument("--min-year", type=int, help="只保留该年份及之后的文献")
    g.add_argument("--top-fulltext", type=int, help="下载解析全文的 Top-N 篇数（默认 8）")
    g.add_argument("--no-fulltext", action="store_true", help="纯摘要模式，完全不下载 PDF")

    g2 = p.add_argument_group("生成")
    g2.add_argument("--provider", choices=["deepseek", "openai", "ollama", "stub"],
                    help="LLM 提供商，覆盖 .env 中的 LLM_PROVIDER")
    g2.add_argument("--clusters", type=int, help="主题簇数量，0=自动推断")
    g2.add_argument("--lang", choices=["zh", "en"], help="综述正文语言")
    g2.add_argument("--critic-rounds", type=int, help="Critic 最多打回补文献的轮数（默认 2）")
    g2.add_argument("--human", action="store_true", help="定稿前挂起等待人工审核")

    g3 = p.add_argument_group("其他")
    g3.add_argument("-o", "--output", help="输出目录（默认 output/）")
    g3.add_argument("--thread-id", default=None, help="检查点线程 ID，同名可断点续跑")
    g3.add_argument("--resume", action="store_true",
                    help="续跑被 --human 挂起的 thread（需配合 --thread-id）")
    g3.add_argument("--feedback", default=None,
                    help="续跑时的人工审核意见（可省略，默认 approve 通过）")
    g3.add_argument("--dry-run", action="store_true",
                    help="离线试跑：不调 LLM（等价 --provider stub）、不下载 PDF")
    g3.add_argument("--print-graph", action="store_true", help="打印状态机结构后退出")
    g3.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 日志")
    return p


def _apply_overrides(args: argparse.Namespace) -> None:
    """把命令行参数写回环境变量，再让 Settings 重新加载（保持单一配置源）。"""
    mapping = {
        "ENABLED_SOURCES": args.sources,
        "TARGET_PAPER_COUNT": args.target,
        "MAX_RESULTS_PER_QUERY": args.per_query,
        "MIN_YEAR": args.min_year,
        "TOP_N_FULLTEXT": args.top_fulltext,
        "LLM_PROVIDER": args.provider,
        "N_CLUSTERS": args.clusters,
        "REPORT_LANGUAGE": args.lang,
        "MAX_CRITIC_ROUNDS": args.critic_rounds,
        "OUTPUT_DIR": args.output,
    }
    for key, value in mapping.items():
        if value is not None:
            os.environ[key] = str(value)

    if args.no_fulltext or args.dry_run:
        os.environ["ENABLE_FULLTEXT"] = "false"
    if args.dry_run:
        os.environ["LLM_PROVIDER"] = "stub"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    _apply_overrides(args)

    from src.config import get_settings

    settings = get_settings(refresh=True)

    if args.print_graph:
        from src.agent.graph import build_graph

        g = build_graph().get_graph()
        try:
            print(g.draw_ascii())
        except ImportError:
            print("(未安装 grandalf，改用 Mermaid 文本；`pip install grandalf` 可看 ASCII 图)\n")
            print(g.draw_mermaid())
        return 0

    if not args.topic and not args.resume:
        parser.print_help()
        return 2

    for warn in settings.validate():
        print(f"[配置提醒] {warn}", file=sys.stderr)
    if not args.resume and settings.llm_provider != "stub" and not settings.model_config().get("api_key"):
        print("\n缺少 LLM API key，无法生成。可先用 --dry-run 走通流程。", file=sys.stderr)
        return 1

    settings.ensure_dirs()
    print("=" * 72)
    print(f"  主题      : {args.topic}")
    print(f"  检索源    : {', '.join(settings.enabled_sources)}")
    print(f"  LLM       : {settings.llm_provider} / {settings.model_config()['model']}")
    print(f"  目标规模  : {settings.target_paper_count} 篇 | 全文 Top-{settings.top_n_fulltext}"
          f"{'（已禁用）' if not settings.enable_fulltext else ''}")
    print(f"  输出目录  : {settings.output_dir}")
    print("=" * 72)

    from src.agent.graph import run_review

    started = time.time()
    try:
        if args.resume:
            if not args.thread_id:
                print("--resume 需要 --thread-id 指定被挂起的运行。", file=sys.stderr)
                return 2
            thread_id = args.thread_id
            final = run_review(
                topic=args.topic or "",
                constraints=args.constraints,
                thread_id=thread_id,
                with_human=True,
                feedback=args.feedback or "approve",
                stream=True,
            )
        elif args.human:
            thread_id = args.thread_id or f"run-{int(time.time())}"
            final = run_review(
                topic=args.topic,
                constraints=args.constraints,
                thread_id=thread_id,
                with_human=True,
                stream=True,
            )
        else:
            thread_id = args.thread_id or f"run-{int(time.time())}"
            final = run_review(
                topic=args.topic,
                constraints=args.constraints,
                thread_id=thread_id,
                with_human=False,
                stream=True,
            )
    except RuntimeError as e:
        if "langgraph-checkpoint-sqlite" in str(e):
            print("\n[错误]", e, file=sys.stderr)
            return 1
        raise
    except KeyboardInterrupt:
        print(f"\n已中断。使用 --thread-id {thread_id} --resume 可从检查点续跑。", file=sys.stderr)
        return 130

    elapsed = time.time() - started

    if final.get("interrupted"):
        print("\n[人工审核] 已挂起。运行 "
              f"`python -m src.main --resume --thread-id {thread_id}` 完成定稿。")
        return 0

    artifacts = final.get("artifacts") or {}
    print("\n" + "=" * 72)
    if not artifacts:
        print("未生成成稿。请检查上方日志（常见原因：检索为空 / LLM 调用失败）。")
        return 1

    print(f"完成，用时 {elapsed:.0f}s")
    print(f"  文献池    : {len(final.get('papers') or [])} 篇，"
          f"引用 {len(final.get('citation_map') or {})} 篇")
    print(f"  主题小节  : {len(final.get('sections') or {})} 个")
    print(f"  研究空白  : {len(final.get('gaps') or [])} 条")
    for name, path in artifacts.items():
        print(f"  {name:10}: {Path(path).resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
