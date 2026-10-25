"""Command-line interface: ``tounum <command> [options]``.

Examples::

    tounum check-data
    tounum classify train
    tounum denoise train
    tounum denoise compare
    tounum caption train
    tounum pipeline run --input scans/ --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tounum.config import Config, ConfigError, load_config
from tounum.utils.logging_config import setup_logging
from tounum.utils.runs import create_run_dir, latest_run, write_json


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    parser.add_argument("--log-level", default="INFO")


def _new_run(config: Config, task: str, args) -> Path:
    run_dir = create_run_dir(config.path("artifacts"), task, getattr(args, "run_name", None))
    write_json(run_dir / "config_used.json", config.to_dict())
    return run_dir


def _resolve_run(config: Config, explicit: str | None, configured: str | None, task: str) -> Path:
    if explicit:
        return config.resolve(explicit)
    if configured:
        return config.resolve(configured)
    return latest_run(config.path("artifacts"), task)


# -- command handlers ---------------------------------------------------------------
def cmd_check_data(config: Config, args) -> int:
    from tounum.data_checks import check_datasets

    all_ok = True
    for name, ok, message in check_datasets(config):
        all_ok &= ok
        print(f"[{'OK' if ok else 'MISSING'}] {name}: {message}")
    return 0 if all_ok else 1


def cmd_classify_train(config: Config, args) -> int:
    from tounum.classification.train import train_classifier

    train_classifier(config, _new_run(config, "classification", args))
    return 0


def cmd_classify_predict(config: Config, args) -> int:
    from tounum.classification.predict import PhotoClassifier

    run = _resolve_run(config, args.run, config.inference.classifier_run, "classification")
    for result in PhotoClassifier.from_run(run).classify(args.images):
        label = "Photo" if result.is_photo else "Other"
        print(f"{result.path}\t{label}\tP(photo)={result.probability_photo:.4f}")
    return 0


def cmd_denoise_train(config: Config, args) -> int:
    from tounum.denoising.train import train_denoiser

    train_denoiser(config, _new_run(config, "denoising", args))
    return 0


def cmd_denoise_compare(config: Config, args) -> int:
    from tounum.denoising.train import compare_architectures

    print(compare_architectures(config, _new_run(config, "denoising_comparison", args)).round(3))
    return 0


def cmd_denoise_apply(config: Config, args) -> int:
    import numpy as np
    from PIL import Image

    from tounum.denoising.predict import Denoiser

    run = _resolve_run(config, args.run, config.inference.denoiser_run, "denoising")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists (use --overwrite).")
    with Image.open(args.image) as image:
        denoised = Denoiser.from_run(run, config.inference.max_denoise_side).denoise(
            np.asarray(image.convert("RGB"))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(denoised).save(output)
    print(f"Denoised image written to {output}")
    return 0


def cmd_caption_train(config: Config, args) -> int:
    from tounum.captioning.train import train_captioner

    train_captioner(config, _new_run(config, "captioning", args))
    return 0


def cmd_caption_predict(config: Config, args) -> int:
    from tounum.captioning.predict import Captioner

    run = _resolve_run(config, args.run, config.inference.captioner_run, "captioning")
    for path, caption in zip(args.images, Captioner.from_run(run).caption_paths(args.images), strict=True):
        print(f"{path}\t{caption}")
    return 0


def cmd_pipeline_run(config: Config, args) -> int:
    from tounum.captioning.predict import Captioner
    from tounum.classification.predict import PhotoClassifier
    from tounum.denoising.predict import Denoiser
    from tounum.pipeline.inference import run_pipeline

    inference = config.inference
    classifier = PhotoClassifier.from_run(
        _resolve_run(config, args.classifier_run, inference.classifier_run, "classification")
    )
    denoiser = Denoiser.from_run(
        _resolve_run(config, args.denoiser_run, inference.denoiser_run, "denoising"),
        inference.max_denoise_side,
    )
    captioner = Captioner.from_run(
        _resolve_run(config, args.captioner_run, inference.captioner_run, "captioning")
    )
    run_pipeline(Path(args.input), Path(args.output), classifier, denoiser, captioner, args.overwrite)
    return 0


def cmd_import_legacy(config: Config, args) -> int:
    from tounum.pipeline import legacy

    if not (args.classifier_dir or args.denoiser_file or args.captioner_dir):
        raise ValueError("Give at least one of --classifier-dir, --denoiser-file, --captioner-dir.")
    artifacts = config.path("artifacts")
    if args.classifier_dir:
        run = legacy.import_classifier(
            args.classifier_dir, create_run_dir(artifacts, "classification", "legacy")
        )
        print(f"Classifier imported to {run}")
    if args.denoiser_file:
        run = legacy.import_denoiser(args.denoiser_file, create_run_dir(artifacts, "denoising", "legacy"))
        print(f"Denoiser imported to {run}")
    if args.captioner_dir:
        run = legacy.import_captioner(args.captioner_dir, create_run_dir(artifacts, "captioning", "legacy"))
        print(f"Captioner imported to {run}")
    return 0


# -- parser ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tounum", description="TouNum vision pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-data", help="check that the datasets are in place")
    _add_common(p)
    p.set_defaults(handler=cmd_check_data)

    groups = {
        "classify": ("photo / other classifier", "classification"),
        "denoise": ("denoising autoencoder", "denoising"),
        "caption": ("image captioning", "captioning"),
    }
    handlers = {
        ("classify", "train"): cmd_classify_train,
        ("denoise", "train"): cmd_denoise_train,
        ("caption", "train"): cmd_caption_train,
    }
    group_parsers = {}
    for group, (help_text, _) in groups.items():
        group_parser = sub.add_parser(group, help=help_text)
        group_parsers[group] = group_parser.add_subparsers(dest="action", required=True)
        p = group_parsers[group].add_parser("train", help="train and evaluate (writes a new run folder)")
        _add_common(p)
        p.add_argument("--run-name", default=None, help="run folder name (default: timestamp)")
        p.set_defaults(handler=handlers[(group, "train")])

    p = group_parsers["classify"].add_parser("predict", help="classify image files")
    _add_common(p)
    p.add_argument("images", nargs="+")
    p.add_argument("--run", default=None, help="classification run folder (default: latest)")
    p.set_defaults(handler=cmd_classify_predict)

    p = group_parsers["denoise"].add_parser("compare", help="compare the 4 autoencoder architectures")
    _add_common(p)
    p.add_argument("--run-name", default=None)
    p.set_defaults(handler=cmd_denoise_compare)

    p = group_parsers["denoise"].add_parser("apply", help="denoise one image")
    _add_common(p)
    p.add_argument("image")
    p.add_argument("--output", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(handler=cmd_denoise_apply)

    p = group_parsers["caption"].add_parser("predict", help="caption image files")
    _add_common(p)
    p.add_argument("images", nargs="+")
    p.add_argument("--run", default=None)
    p.set_defaults(handler=cmd_caption_predict)

    pipeline = sub.add_parser("pipeline", help="end-to-end inference")
    pipeline_sub = pipeline.add_subparsers(dest="action", required=True)
    p = pipeline_sub.add_parser("run", help="classify, denoise and caption a folder of images")
    _add_common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--classifier-run", default=None)
    p.add_argument("--denoiser-run", default=None)
    p.add_argument("--captioner-run", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(handler=cmd_pipeline_run)

    p = sub.add_parser("import-legacy", help="import models saved by the original notebooks")
    _add_common(p)
    p.add_argument("--classifier-dir", type=Path, help="folder livrable1_artifacts/")
    p.add_argument("--denoiser-file", type=Path, help="file debruiteur_final.keras")
    p.add_argument("--captioner-dir", type=Path, help="folder livrable3_artifacts/")
    p.set_defaults(handler=cmd_import_legacy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.log_level)
    try:
        config = load_config(args.config)
        return int(args.handler(config, args))
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
