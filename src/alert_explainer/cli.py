"""Command line: ``uv run ae <command>``. Run ``uv run ae --help`` for the list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alert_explainer.config import get_settings


def _cmd_etl(raw: Path, processed: Path, reports: Path, docs: Path | None) -> None:
    from alert_explainer.data.etl import run_etl

    print(f"ETL: reading CSVs under {raw}")
    stats = run_etl(raw, processed, reports, docs)
    t = stats["totals"]
    print(
        f"Done: {t['rows_read']:,} read → {t['rows_final']:,} final rows, "
        f"{stats['n_features']} features, {t['duplicates_removed']:,} duplicates removed, "
        f"Parquet loads in {stats['parquet_load_seconds']} s"
    )
    print(json.dumps(stats["families_final"], indent=2))


def _cmd_split(processed: Path, reports: Path, seed: int) -> None:
    from alert_explainer.data.splits import run_splits

    summary = run_splits(processed, reports, seed)
    print(json.dumps(summary["family_by_split"], indent=2))
    shared = summary["attack_families_in_both_day_splits"]
    print(f"Attack families in both Mon–Wed and Thu–Fri: {shared or 'none'}")


def _torch_args(p: argparse.ArgumentParser, default_device: str) -> None:
    p.add_argument(
        "--device",
        default=default_device,
        help="auto | cpu | mps | cuda (default: AE_DEVICE or auto; run `ae bench-mlp` to choose)",
    )
    p.add_argument("--max-epochs", type=int, default=None, help="override the config's max epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override the config's batch size")


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(prog="ae", description="Alert Explainer pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("etl", help="Day 2: raw CSVs → data/processed/flows.parquet + data card")
    p.add_argument("--raw-dir", type=Path, default=s.raw_dir)

    sub.add_parser("split", help="Day 3: stratified 70/15/15 + cross-day split")

    p = sub.add_parser("train-lgbm", help="Day 3: LightGBM baseline with calibration + threshold")
    p.add_argument("--grid", choices=["quick", "default"], default="default")
    p.add_argument(
        "--grid-benign-cap",
        type=int,
        default=300_000,
        help="BENIGN rows sampled for the grid search (all attack rows are kept); 0 = use all",
    )
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--max-rounds", type=int, default=2000)
    p.add_argument("--patience", type=int, default=50)

    p = sub.add_parser("bench-mlp", help="Day 4: time MLP training epochs on CPU vs MPS/CUDA")
    p.add_argument("--rows", type=int, default=300_000)
    p.add_argument("--epochs", type=int, default=2)

    p = sub.add_parser(
        "train-mlp", help="Days 4–5: PyTorch MLP (grid + focal-loss ablation) → models/mlp"
    )
    p.add_argument(
        "--grid",
        choices=["none", "quick", "default"],
        default="default",
        help="none = default config only (Day 4); default = 7-config search (Day 5)",
    )
    p.add_argument("--grid-benign-cap", type=int, default=300_000)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--no-ablation", action="store_true", help="skip the focal-loss final run")
    _torch_args(p, s.device)

    p = sub.add_parser(
        "cross-day", help="Day 5: train Mon–Wed, test unseen Thu–Fri attacks (LGBM + MLP)"
    )
    p.add_argument("--models", default="lgbm,mlp", help="comma-separated: lgbm, mlp")
    p.add_argument("--target-fpr", type=float, default=0.01)
    _torch_args(p, s.device)

    sub.add_parser("compare", help="Day 5: reports/model_comparison.md + models/*/model_card.md")

    p = sub.add_parser("make-synthetic", help="Write small fake CSVs in TrafficLabelling layout")
    p.add_argument("--out", type=Path, default=s.data_dir / "demo" / "raw")
    p.add_argument("--rows", type=int, default=2000)

    p = sub.add_parser("demo", help="Whole pipeline on synthetic data under data/demo/")
    p.add_argument("--out", type=Path, default=s.data_dir / "demo")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--device", default=s.device)

    args = parser.parse_args(argv)

    if args.command == "etl":
        _cmd_etl(args.raw_dir, s.processed_dir, s.reports_dir, s.docs_dir)
    elif args.command == "split":
        _cmd_split(s.processed_dir, s.reports_dir, s.seed)
    elif args.command == "train-lgbm":
        from alert_explainer.detect.lgbm import train_lgbm

        train_lgbm(
            s.processed_dir,
            s.models_dir,
            s.reports_dir,
            grid=args.grid,
            grid_benign_cap=args.grid_benign_cap,
            target_fpr=args.target_fpr,
            seed=s.seed,
            max_rounds=args.max_rounds,
            patience=args.patience,
        )
    elif args.command == "bench-mlp":
        from alert_explainer.detect.train_mlp import bench_devices

        bench_devices(s.processed_dir, rows=args.rows, epochs=args.epochs, seed=s.seed)
    elif args.command == "train-mlp":
        from alert_explainer.detect.train_mlp import train_mlp

        train_mlp(
            s.processed_dir,
            s.models_dir,
            s.reports_dir,
            grid=args.grid,
            grid_benign_cap=args.grid_benign_cap,
            target_fpr=args.target_fpr,
            seed=s.seed,
            device=args.device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            ablation=not args.no_ablation,
        )
    elif args.command == "cross-day":
        from alert_explainer.detect.cross_day import run_cross_day

        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        unknown = set(models) - {"lgbm", "mlp"}
        if unknown:
            parser.error(f"unknown model(s) {sorted(unknown)}; use lgbm and/or mlp")
        run_cross_day(
            s.processed_dir,
            s.models_dir,
            s.reports_dir,
            models=models,
            target_fpr=args.target_fpr,
            seed=s.seed,
            device=args.device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
    elif args.command == "compare":
        from alert_explainer.detect.compare import write_comparison

        write_comparison(s.models_dir, s.reports_dir)
    elif args.command == "make-synthetic":
        from alert_explainer.data.synthetic import make_synthetic

        paths = make_synthetic(args.out, args.rows)
        print(f"Wrote {len(paths)} synthetic CSVs to {args.out}")
    elif args.command == "demo":
        from alert_explainer.data.synthetic import make_synthetic
        from alert_explainer.detect.compare import write_comparison
        from alert_explainer.detect.cross_day import run_cross_day
        from alert_explainer.detect.lgbm import train_lgbm
        from alert_explainer.detect.train_mlp import train_mlp

        out: Path = args.out
        make_synthetic(out / "raw", args.rows)
        _cmd_etl(out / "raw", out / "processed", out / "reports", out / "docs")
        _cmd_split(out / "processed", out / "reports", s.seed)
        train_lgbm(out / "processed", out / "models", out / "reports", grid="quick", seed=s.seed)
        train_mlp(
            out / "processed", out / "models", out / "reports", grid="quick",
            grid_benign_cap=1000, seed=s.seed, device=args.device, max_epochs=5, batch_size=256,
        )  # fmt: skip
        run_cross_day(
            out / "processed", out / "models", out / "reports", seed=s.seed,
            device=args.device, max_epochs=5, batch_size=256,
        )  # fmt: skip
        write_comparison(out / "models", out / "reports")
        print(f"Demo outputs in {out}/ (synthetic data: the metrics mean nothing)")


if __name__ == "__main__":
    main()
