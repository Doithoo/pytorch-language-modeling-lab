"""Command-line interface for data, training, evaluation, and generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .config import config_to_dict, load_config, parse_overrides
from .data import inspect_prepared_data, prepare_data
from .evaluation import compare_runs, evaluate_checkpoint
from .inference import Predictor
from .models import list_models
from .preflight import print_preflight
from .training.train import run_training
from .utils import write_json


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="override one YAML value; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lm", description="Educational causal language modeling lab")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="normalize, split, tokenize, and version a corpus")
    _common_config(prepare)
    prepare.add_argument("--overwrite", action="store_true")

    inspect = subparsers.add_parser("inspect-data", help="verify prepared files and print corpus statistics")
    inspect.add_argument("--manifest", required=True, type=Path)
    inspect.add_argument("--processed-root", type=Path)
    inspect.add_argument("--no-verify-hashes", action="store_true")

    train = subparsers.add_parser("train", help="train or resume a language model")
    _common_config(train)
    train.add_argument("--manifest", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a prepared split")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--split", choices=["train", "valid", "test"], default="test")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--processed-root", type=Path)
    evaluate.add_argument("--output", type=Path)

    generate = subparsers.add_parser("generate", help="generate text from an embedded-tokenizer checkpoint")
    generate.add_argument("--checkpoint", required=True, type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    generate.add_argument("--max-new-tokens", type=int)
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--top-p", type=float)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--no-cache", action="store_true")
    generate.add_argument("--output", type=Path)

    compare = subparsers.add_parser("compare-runs", help="compare metrics from completed run directories")
    compare.add_argument("--run", action="append", required=True, type=Path)
    compare.add_argument("--output", type=Path)

    subparsers.add_parser("list-models", help="show built-in model names and contracts")
    show = subparsers.add_parser("show-config", help="print a fully resolved YAML configuration as JSON")
    _common_config(show)
    preflight = subparsers.add_parser("preflight", help="print local or Kaggle environment information")
    preflight.add_argument("--work-dir", type=Path, default=Path("."))
    return parser


def _load_from_args(args: argparse.Namespace):
    return load_config(args.config, parse_overrides(args.set))


def _generation_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    overrides = {key: value for key, value in values.items() if value is not None}
    if args.no_cache:
        overrides["use_cache"] = False
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-data":
            cfg = _load_from_args(args)
            manifest = prepare_data(cfg, overwrite=args.overwrite)
            print(json.dumps({"status": "completed", "manifest": str(manifest)}))
        elif args.command == "inspect-data":
            inspection = inspect_prepared_data(
                args.manifest,
                verify_hashes=not args.no_verify_hashes,
                processed_root=args.processed_root,
            )
            print(json.dumps(inspection, ensure_ascii=False, indent=2))
        elif args.command == "train":
            cfg = _load_from_args(args)
            if args.device:
                cfg = replace(cfg, runtime=replace(cfg.runtime, device=args.device))
            run_dir = run_training(
                cfg,
                manifest_path=args.manifest,
                resume_path=args.resume,
                dry_run=args.dry_run,
            )
            if run_dir is not None:
                print(json.dumps({"status": "completed", "run_dir": str(run_dir)}))
        elif args.command == "evaluate":
            evaluation = evaluate_checkpoint(
                args.checkpoint,
                args.manifest,
                split=args.split,
                device=args.device,
                max_batches=args.max_batches,
                processed_root=args.processed_root,
            )
            if args.output:
                write_json(args.output, evaluation)
            print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        elif args.command == "generate":
            predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
            generation = predictor.generate(args.prompt, **_generation_overrides(args))
            if args.output:
                write_json(args.output, generation)
            print(json.dumps(generation, ensure_ascii=False, indent=2))
        elif args.command == "compare-runs":
            comparison = compare_runs(args.run)
            if args.output:
                write_json(args.output, comparison)
            print(json.dumps(comparison, ensure_ascii=False, indent=2))
        elif args.command == "list-models":
            print(json.dumps(list_models(), indent=2))
        elif args.command == "show-config":
            print(json.dumps(config_to_dict(_load_from_args(args)), ensure_ascii=False, indent=2))
        elif args.command == "preflight":
            print_preflight(args.work_dir)
        else:  # pragma: no cover
            parser.error(f"unknown command: {args.command}")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
