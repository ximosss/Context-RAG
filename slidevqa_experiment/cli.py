from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .clients import RuntimeClients, WeaveTracker
from .config import Settings
from .pipeline import build_offline, run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SlideVQA multimodal RAG experiment runner (dataset controlled by DATASET_NAME/DATASET_DIR)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build offline indices for one variant")
    build_parser.add_argument("--variant", choices=["baseline", "enhanced"], required=True)
    build_parser.add_argument("--max-samples", type=int, default=None)
    build_parser.add_argument("--rebuild", action="store_true")
    build_parser.add_argument("--corpus-limit", type=int, default=None)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation for one variant")
    eval_parser.add_argument("--variant", choices=["baseline", "enhanced"], required=True)
    eval_parser.add_argument("--max-samples", type=int, default=None)
    eval_parser.add_argument("--run-name", default=None)
    eval_parser.add_argument(
        "--with-ragas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RAGAS evaluation (default: enabled).",
    )

    run_parser = subparsers.add_parser("run", help="Build + eval for one variant")
    run_parser.add_argument("--variant", choices=["baseline", "enhanced"], required=True)
    run_parser.add_argument("--max-samples", type=int, default=None)
    run_parser.add_argument("--rebuild", action="store_true")
    run_parser.add_argument("--corpus-limit", type=int, default=None)
    run_parser.add_argument("--run-name", default=None)
    run_parser.add_argument(
        "--with-ragas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RAGAS evaluation (default: enabled).",
    )

    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    settings.ensure_dirs()

    tracker = WeaveTracker(settings)
    runtime = RuntimeClients(settings)

    try:
        if args.command == "build":
            summary = await build_offline(
                runtime,
                settings,
                tracker,
                variant=args.variant,
                max_samples=args.max_samples,
                rebuild=args.rebuild,
                corpus_limit=args.corpus_limit,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "eval":
            summary = await run_eval(
                runtime,
                settings,
                tracker,
                variant=args.variant,
                max_samples=args.max_samples,
                run_name=args.run_name,
                with_ragas=args.with_ragas,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.command == "run":
            build_summary = await build_offline(
                runtime,
                settings,
                tracker,
                variant=args.variant,
                max_samples=args.max_samples,
                rebuild=args.rebuild,
                corpus_limit=args.corpus_limit,
            )
            eval_summary = await run_eval(
                runtime,
                settings,
                tracker,
                variant=args.variant,
                max_samples=args.max_samples,
                run_name=args.run_name,
                with_ragas=args.with_ragas,
            )
            print(json.dumps({"build": build_summary, "eval": eval_summary}, ensure_ascii=False, indent=2))
            return 0

        return 1
    finally:
        await runtime.close()


def main() -> int:
    return asyncio.run(_main())
