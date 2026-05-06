"""Few-shot linear-probe training and evaluation on frozen coarse-vs-fine SAM features."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from rwtd_sam3.eval.coarse_vs_fine_sam_scales import (
    ARCHITEXTURE_COARSE_VS_FINE_OUTPUT_ROOT,
    CSTD_COARSE_VS_FINE_OUTPUT_ROOT,
    COARSE_VS_FINE_OUTPUT_ROOT,
    EXPERIMENT_CONTRACT_PATH,
    FeatureSampleSource,
    _load_feature_samples,
    _load_stage2_raw_samples,
    _resolve_torch_device,
    iter_feature_bundles,
    prepare_coarse_vs_fine_output_dir,
    resolve_stage2_dataset_id,
)
from rwtd_sam3.eval.experiment_terms import (
    ExperimentTermsSection,
    format_shape_mapping,
    render_experiment_terms_markdown,
    render_relative_path,
)
from rwtd_sam3.eval.metrics import (
    CANONICAL_EVALUATION_CONTRACT,
    CANONICAL_PRIMARY_METRIC,
    CANONICAL_SECONDARY_METRIC,
    build_canonical_evaluation_fields,
)
from rwtd_sam3.eval.runner import (
    SampleEvaluationResult,
    build_visual_record,
    discovered_package_versions,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from rwtd_sam3.models.sam3_coarse_vs_fine_linear_probe import (
    COARSE_VS_FINE_LINEAR_PROBE_SETTINGS,
    LINEAR_PROBE_VARIANT_SPECS,
    LinearProbeLossResult,
    LinearProbeOutput,
    LinearSegmentationProbeModule,
    compute_linear_probe_loss,
    evaluate_linear_probe_partition,
    predict_linear_probe_label_map,
    resolve_linear_probe_variant_levels,
)
from rwtd_sam3.models.sam3_coarse_vs_fine_scale_probe import (
    CoarseVsFineScaleSample,
    Sam3CoarseVsFineScaleFeatureExtractor,
    Sam3CoarseVsFineScaleProbeRuntimeError,
    materialize_feature_levels_on_device,
)
from rwtd_sam3.utils.visualization import save_prediction_panel


LOGGER = logging.getLogger(__name__)
RWTD_LINEAR_PROBE_OUTPUT_ROOT = COARSE_VS_FINE_OUTPUT_ROOT.parent / "linear_probe"
ARCHITEXTURE_LINEAR_PROBE_OUTPUT_ROOT = ARCHITEXTURE_COARSE_VS_FINE_OUTPUT_ROOT.parent / "linear_probe"
CSTD_LINEAR_PROBE_OUTPUT_ROOT = CSTD_COARSE_VS_FINE_OUTPUT_ROOT.parent / "linear_probe"


@dataclass(frozen=True)
class LinearProbeTrainingArtifacts:
    checkpoint_path: Path
    initial_level_weight_norms: dict[str, float]
    final_level_weight_norms: dict[str, float]
    training_history_rows: list[dict[str, Any]]
    level_input_dims: dict[str, int]


@dataclass(frozen=True)
class LinearProbeEvalBundle:
    row: dict[str, Any]
    metric_summary: str
    prediction_a: np.ndarray
    prediction_b: np.ndarray


def resolve_linear_probe_output_root(args) -> Path:
    dataset_source = str(getattr(args, "dataset_source", "rwtd"))
    if dataset_source == "architexture_binary":
        return ARCHITEXTURE_LINEAR_PROBE_OUTPUT_ROOT
    if dataset_source == "cstd_binary":
        return CSTD_LINEAR_PROBE_OUTPUT_ROOT
    return RWTD_LINEAR_PROBE_OUTPUT_ROOT


def prepare_linear_probe_output_dir(
    output_dir: str | None,
    *,
    args,
    variant: str,
    run_kind: str,
    split: str | None,
) -> Path:
    if output_dir is None:
        suffix = f"{run_kind}_{variant}"
        if str(getattr(args, "dataset_source", "rwtd")) == "architexture_binary":
            suffix = f"{suffix}_{getattr(args, 'route', None)}"
        if split is not None:
            suffix = f"{suffix}_{split}"
        return resolve_linear_probe_output_root(args) / suffix
    return Path(output_dir)


def run_coarse_vs_fine_linear_probe_train(args) -> dict[str, Any]:
    """Train one few-shot linear probe and immediately evaluate it."""

    output_dir = prepare_linear_probe_output_dir(
        args.output_dir,
        args=args,
        variant=args.variant,
        run_kind="train",
        split=args.eval_split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_linear_probe_run_config(args, run_kind="train")
    write_json(output_dir / "config.json", config)

    extractor = Sam3CoarseVsFineScaleFeatureExtractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token,
        official_checkpoint_path=args.official_checkpoint_path,
    )
    selected_level_names = discover_linear_probe_selected_levels(
        extractor=extractor,
        dataset_source=str(getattr(args, "dataset_source", "rwtd")),
        dataset_id=getattr(args, "dataset_id", None),
        cache_dir=args.cache_dir,
        route=getattr(args, "route", None),
        benchmark_root=getattr(args, "benchmark_root", None),
        dataset_root=getattr(args, "dataset_root", None),
        split=args.train_split,
        limit=args.train_limit,
        variant=args.variant,
    )
    train_samples = _load_feature_samples(
        extractor=extractor,
        dataset_source=str(getattr(args, "dataset_source", "rwtd")),
        dataset_id=getattr(args, "dataset_id", None),
        cache_dir=args.cache_dir,
        route=getattr(args, "route", None),
        benchmark_root=getattr(args, "benchmark_root", None),
        dataset_root=getattr(args, "dataset_root", None),
        split=args.train_split,
        limit=args.train_limit,
        variant="fpn_2_only",
        expected_selected_level_names=selected_level_names,
    )
    eval_samples = _load_feature_samples(
        extractor=extractor,
        dataset_source=str(getattr(args, "dataset_source", "rwtd")),
        dataset_id=getattr(args, "dataset_id", None),
        cache_dir=args.cache_dir,
        route=getattr(args, "route", None),
        benchmark_root=getattr(args, "benchmark_root", None),
        dataset_root=getattr(args, "dataset_root", None),
        split=args.eval_split,
        limit=args.eval_limit,
        variant="fpn_2_only",
        expected_selected_level_names=selected_level_names,
    )

    reference_bundle = train_samples.reference_bundle
    write_text(
        output_dir / "experiment_terms.md",
        build_linear_probe_experiment_terms_markdown(
            args,
            run_kind="train",
            selected_level_names=reference_bundle.selected_level_names,
            raw_level_shapes=reference_bundle.raw_level_shapes,
            aligned_level_shapes=reference_bundle.aligned_level_shapes,
            coarsest_level_name=reference_bundle.coarsest_level_name,
            coarsest_grid_size=reference_bundle.coarsest_grid_size,
            train_sample_count=len(train_samples.samples),
            eval_sample_count=len(eval_samples.samples),
            materialization_mode=train_samples.materialization_mode,
        ),
    )
    probe_module, training_artifacts = train_linear_probe(
        train_samples=train_samples,
        output_dir=output_dir,
        args=args,
    )
    return evaluate_linear_probe(
        probe_module=probe_module,
        eval_samples=eval_samples,
        output_dir=output_dir,
        args=args,
        checkpoint_path=training_artifacts.checkpoint_path,
        checkpoint_metadata={
            "variant": args.variant,
            "selected_level_names": list(reference_bundle.selected_level_names),
            "coarsest_level_name": reference_bundle.coarsest_level_name,
            "level_input_dims": training_artifacts.level_input_dims,
            "post_concat_normalization": bool(args.post_concat_normalization),
            "initial_level_weight_norms": training_artifacts.initial_level_weight_norms,
            "final_level_weight_norms": training_artifacts.final_level_weight_norms,
            "train_history_rows": training_artifacts.training_history_rows,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "seed": int(getattr(args, "seed", 0)),
            "materialization_mode": train_samples.materialization_mode,
        },
        run_kind="train",
    )


def run_coarse_vs_fine_linear_probe_eval(args) -> dict[str, Any]:
    """Evaluate one saved few-shot linear probe checkpoint."""

    import torch

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Linear-probe checkpoint path does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_variant = str(checkpoint["variant"])
    setattr(args, "variant", checkpoint_variant)

    output_dir = prepare_linear_probe_output_dir(
        args.output_dir,
        args=args,
        variant=checkpoint_variant,
        run_kind="eval",
        split=args.split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_linear_probe_run_config(args, run_kind="eval")
    config["checkpoint_path"] = str(checkpoint_path)
    config["variant"] = checkpoint_variant
    write_json(output_dir / "config.json", config)

    extractor = Sam3CoarseVsFineScaleFeatureExtractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token,
        official_checkpoint_path=args.official_checkpoint_path,
    )
    eval_samples = _load_feature_samples(
        extractor=extractor,
        dataset_source=str(getattr(args, "dataset_source", "rwtd")),
        dataset_id=getattr(args, "dataset_id", None),
        cache_dir=args.cache_dir,
        route=getattr(args, "route", None),
        benchmark_root=getattr(args, "benchmark_root", None),
        dataset_root=getattr(args, "dataset_root", None),
        split=args.split,
        limit=args.limit,
        variant="fpn_2_only",
        expected_selected_level_names=tuple(checkpoint["selected_level_names"]),
    )
    reference_bundle = eval_samples.reference_bundle
    write_text(
        output_dir / "experiment_terms.md",
        build_linear_probe_experiment_terms_markdown(
            args,
            run_kind="eval",
            selected_level_names=reference_bundle.selected_level_names,
            raw_level_shapes=reference_bundle.raw_level_shapes,
            aligned_level_shapes=reference_bundle.aligned_level_shapes,
            coarsest_level_name=reference_bundle.coarsest_level_name,
            coarsest_grid_size=reference_bundle.coarsest_grid_size,
            train_sample_count=None,
            eval_sample_count=len(eval_samples.samples),
            materialization_mode=eval_samples.materialization_mode,
            checkpoint_path=checkpoint_path,
        ),
    )
    probe_module = _build_linear_probe_module_from_checkpoint(checkpoint)
    probe_module.load_state_dict(checkpoint["probe_state_dict"])
    return evaluate_linear_probe(
        probe_module=probe_module,
        eval_samples=eval_samples,
        output_dir=output_dir,
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint,
        run_kind="eval",
    )


def discover_linear_probe_selected_levels(
    *,
    extractor: Sam3CoarseVsFineScaleFeatureExtractor,
    dataset_source: str,
    dataset_id: str | None,
    cache_dir: str | None,
    route: str | None,
    benchmark_root: str | None,
    dataset_root: str | None,
    split: str,
    limit: int | None,
    variant: str,
) -> tuple[str, ...]:
    """Discover available SAM pyramid levels and resolve the requested linear-probe subset."""

    _, samples = _load_stage2_raw_samples(
        dataset_source=dataset_source,
        dataset_id=dataset_id,
        cache_dir=cache_dir,
        route=route,
        benchmark_root=benchmark_root,
        dataset_root=dataset_root,
        split=split,
        limit=limit,
    )
    if not samples:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"No samples were loaded for dataset_source={dataset_source} split '{split}' with limit={limit}."
        )
    _, first_pyramid = extractor.extract_sam_pyramid(samples[0].image)
    return resolve_linear_probe_variant_levels(
        variant=variant,
        available_level_names=tuple(first_pyramid.keys()),
    )


def train_linear_probe(
    *,
    train_samples: FeatureSampleSource,
    output_dir: Path,
    args,
) -> tuple[Any, LinearProbeTrainingArtifacts]:
    """Optimize the few-shot linear probe on frozen aligned SAM features."""

    import torch

    if not train_samples.samples:
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Linear probe received an empty train split.")
    reference_bundle = train_samples.reference_bundle
    probe_module = _build_linear_probe_module_from_sample(reference_bundle, args=args)
    train_device = _resolve_torch_device(args.device)
    probe_module.to(train_device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in probe_module.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    initial_level_weight_norms = dict(probe_module.current_level_weight_norms())
    history_rows: list[dict[str, Any]] = []

    LOGGER.info(
        "Few-shot linear probe training start | variant=%s levels=%s coarsest=%s aligned_shapes=%s materialization_mode=%s post_concat_norm=%s optimizer=AdamW lr=%.6f wd=%.6f epochs=%d metrics=%s,%s",
        args.variant,
        list(reference_bundle.selected_level_names),
        reference_bundle.coarsest_level_name,
        reference_bundle.aligned_level_shapes,
        train_samples.materialization_mode,
        bool(args.post_concat_normalization),
        float(args.learning_rate),
        float(args.weight_decay),
        int(args.num_epochs),
        CANONICAL_PRIMARY_METRIC,
        CANONICAL_SECONDARY_METRIC,
    )

    for epoch in range(int(args.num_epochs)):
        probe_module.train()
        epoch_losses: list[float] = []
        epoch_valid_pixels: list[int] = []
        for sample_bundle in iter_feature_bundles(train_samples):
            feature_levels = materialize_feature_levels_on_device(
                sample_bundle.aligned_feature_levels,
                device=train_device,
            )
            optimizer.zero_grad(set_to_none=True)
            output: LinearProbeOutput = probe_module(feature_levels)
            loss_result: LinearProbeLossResult = compute_linear_probe_loss(
                output.logits,
                sample_bundle.coarsest_grid_labels,
            )
            loss_result.loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss_result.loss.detach().cpu().item()))
            epoch_valid_pixels.append(int(loss_result.num_valid_pixels))
        if not epoch_losses:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "Linear probe training produced no valid samples.",
                diagnostics={"variant": args.variant, "split": train_samples.split, "dataset_id": train_samples.dataset_id},
            )
        history_rows.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(mean(epoch_losses)),
                "mean_valid_pixels": float(mean(epoch_valid_pixels)),
            }
        )

    final_level_weight_norms = dict(probe_module.current_level_weight_norms())
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "variant": args.variant,
            "model_id": args.model_id,
            "selected_level_names": list(reference_bundle.selected_level_names),
            "coarsest_level_name": reference_bundle.coarsest_level_name,
            "level_input_dims": {
                level_name: int(reference_bundle.aligned_level_shapes[level_name][0])
                for level_name in reference_bundle.selected_level_names
            },
            "post_concat_normalization": bool(args.post_concat_normalization),
            "probe_state_dict": probe_module.state_dict(),
            "seed": int(getattr(args, "seed", 0)),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "initial_level_weight_norms": initial_level_weight_norms,
            "final_level_weight_norms": final_level_weight_norms,
            "train_history_rows": history_rows,
            "materialization_mode": train_samples.materialization_mode,
        },
        checkpoint_path,
    )
    write_csv(output_dir / "train_history.csv", history_rows)
    write_json(
        output_dir / "weight_norms.json",
        {
            "initial_level_weight_norms": initial_level_weight_norms,
            "final_level_weight_norms": final_level_weight_norms,
        },
    )
    return probe_module, LinearProbeTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        initial_level_weight_norms=initial_level_weight_norms,
        final_level_weight_norms=final_level_weight_norms,
        training_history_rows=history_rows,
        level_input_dims={
            level_name: int(reference_bundle.aligned_level_shapes[level_name][0])
            for level_name in reference_bundle.selected_level_names
        },
    )


def evaluate_linear_probe(
    *,
    probe_module: Any,
    eval_samples: FeatureSampleSource,
    output_dir: Path,
    args,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    run_kind: str,
) -> dict[str, Any]:
    """Evaluate one saved few-shot linear probe checkpoint."""

    import torch

    probe_module.eval()
    eval_device = _resolve_torch_device(args.device)
    probe_module.to(eval_device)
    rows: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []
    for sample_bundle in iter_feature_bundles(eval_samples):
        with torch.no_grad():
            feature_levels = materialize_feature_levels_on_device(
                sample_bundle.aligned_feature_levels,
                device=eval_device,
            )
            output: LinearProbeOutput = probe_module(feature_levels)
            class_map = predict_linear_probe_label_map(output.logits)
        partition = evaluate_linear_probe_partition(sample=sample_bundle.sample, coarsest_grid_label_map=class_map)
        row = build_linear_probe_eval_row(
            sample_bundle=sample_bundle,
            dataset_id=eval_samples.dataset_id,
            dataset_source=eval_samples.dataset_source,
            dataset_route=eval_samples.dataset_route,
            partition=partition,
            variant=str(checkpoint_metadata["variant"]),
            checkpoint_path=checkpoint_path,
            probe_module=probe_module,
            args=args,
        )
        rows.append(row)
        metric_summary = (
            f"Eval mIoU={row['eval_miou']:.3f} "
            f"Eval ARI={row['eval_ari']:.3f} "
            f"Assign={row['assignment_used']} "
            f"W={row['final_level_weight_norms_compact']}"
        )
        result = LinearProbeEvalBundle(
            row=row,
            metric_summary=metric_summary,
            prediction_a=partition.prediction_a,
            prediction_b=partition.prediction_b,
        )
        _save_linear_probe_result_bundle(
            output_dir=output_dir,
            sample_bundle=sample_bundle,
            result=result,
            save_visuals=bool(args.save_visuals),
        )
        if args.save_visuals:
            visual_records.append(
                build_visual_record(
                    sample=sample_bundle.sample,
                    evaluation=SampleEvaluationResult(
                        row=row,
                        metric_summary=metric_summary,
                        prediction_a=partition.prediction_a,
                        prediction_b=partition.prediction_b,
                    ),
                    protocol=f"coarse_vs_fine_linear_probe:{checkpoint_metadata['variant']}",
                    visual_path=Path("visuals") / f"{sample_bundle.sample.index}.png",
                )
            )
    if not rows:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Linear probe evaluation produced no sample rows."
        )
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
    write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)
    summary = build_linear_probe_summary(
        rows=rows,
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        run_kind=run_kind,
    )
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", [flatten_linear_probe_summary(summary)])
    write_text(output_dir / "summary.md", build_linear_probe_summary_markdown(summary))
    return summary


def build_linear_probe_eval_row(
    *,
    sample_bundle: CoarseVsFineScaleSample,
    dataset_id: str,
    dataset_source: str,
    dataset_route: str | None,
    partition,
    variant: str,
    checkpoint_path: Path,
    probe_module: Any,
    args,
) -> dict[str, Any]:
    weight_norms = dict(probe_module.current_level_weight_norms())
    row: dict[str, Any] = {
        "variant": variant,
        "dataset_id": dataset_id,
        "dataset_source": dataset_source,
        "dataset_route": dataset_route,
        "split": sample_bundle.sample.split,
        "sample_index": sample_bundle.sample.index,
        "crop_name": sample_bundle.sample.crop_name,
        "assignment_used": partition.assignment_used,
        "direct_eval_miou": partition.direct_eval_miou,
        "direct_eval_ari": partition.direct_eval_ari,
        "swapped_eval_miou": partition.swapped_eval_miou,
        "swapped_eval_ari": partition.swapped_eval_ari,
        "selected_level_names_json": json.dumps(list(sample_bundle.selected_level_names)),
        "coarsest_level_name": sample_bundle.coarsest_level_name,
        "raw_level_names_json": json.dumps(list(sample_bundle.raw_level_names)),
        "raw_level_shapes_json": json.dumps(sample_bundle.raw_level_shapes, sort_keys=True),
        "aligned_level_shapes_json": json.dumps(sample_bundle.aligned_level_shapes, sort_keys=True),
        "coarsest_grid_height": int(sample_bundle.coarsest_grid_size[0]),
        "coarsest_grid_width": int(sample_bundle.coarsest_grid_size[1]),
        "post_concat_normalization": bool(args.post_concat_normalization),
        "level_weight_norms_json": json.dumps(weight_norms, sort_keys=True),
        "final_level_weight_norms_compact": ",".join(f"{name}={value:.3f}" for name, value in weight_norms.items()),
        "checkpoint_path": str(checkpoint_path),
        "evaluation_view": "permutation_invariant_binary_partition",
    }
    row.update(
        build_canonical_evaluation_fields(
            miou=float(partition.eval_miou),
            ari=float(partition.eval_ari),
            evaluation_view="permutation_invariant_binary_partition",
        )
    )
    return row


def build_linear_probe_summary(
    *,
    rows: list[dict[str, Any]],
    args,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    run_kind: str,
) -> dict[str, Any]:
    mean_metrics = {
        CANONICAL_PRIMARY_METRIC: float(mean(row[CANONICAL_PRIMARY_METRIC] for row in rows)),
        CANONICAL_SECONDARY_METRIC: float(mean(row[CANONICAL_SECONDARY_METRIC] for row in rows)),
    }
    median_metrics = {
        CANONICAL_PRIMARY_METRIC: float(median(row[CANONICAL_PRIMARY_METRIC] for row in rows)),
        CANONICAL_SECONDARY_METRIC: float(median(row[CANONICAL_SECONDARY_METRIC] for row in rows)),
    }
    return {
        "run_kind": run_kind,
        "variant": str(checkpoint_metadata["variant"]),
        "dataset_id": resolve_stage2_dataset_id(args),
        "dataset_source": str(getattr(args, "dataset_source", "rwtd")),
        "dataset_route": getattr(args, "route", None),
        "split": getattr(args, "split", getattr(args, "eval_split", None)),
        "train_split": getattr(args, "train_split", None),
        "model_id": args.model_id,
        "device": args.device,
        "official_checkpoint_path": args.official_checkpoint_path,
        "checkpoint_path": str(checkpoint_path),
        "num_evaluated_samples": len(rows),
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "selected_level_names": list(checkpoint_metadata["selected_level_names"]),
        "coarsest_level_name": checkpoint_metadata["coarsest_level_name"],
        "post_concat_normalization": bool(checkpoint_metadata["post_concat_normalization"]),
        "learning_rate": float(checkpoint_metadata["learning_rate"]),
        "weight_decay": float(checkpoint_metadata["weight_decay"]),
        "num_epochs": int(checkpoint_metadata["num_epochs"]),
        "seed": int(checkpoint_metadata["seed"]),
        "initial_level_weight_norms": dict(checkpoint_metadata["initial_level_weight_norms"]),
        "final_level_weight_norms": dict(checkpoint_metadata["final_level_weight_norms"]),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def flatten_linear_probe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": summary["variant"],
        "split": summary["split"],
        "num_evaluated_samples": summary["num_evaluated_samples"],
        CANONICAL_PRIMARY_METRIC: summary["mean_metrics"][CANONICAL_PRIMARY_METRIC],
        CANONICAL_SECONDARY_METRIC: summary["mean_metrics"][CANONICAL_SECONDARY_METRIC],
        "coarsest_level_name": summary["coarsest_level_name"],
        "selected_level_names_json": json.dumps(summary["selected_level_names"]),
        "final_level_weight_norms_json": json.dumps(summary["final_level_weight_norms"], sort_keys=True),
        "checkpoint_path": summary["checkpoint_path"],
    }


def build_linear_probe_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Coarse-vs-Fine SAM Scales Linear Probe Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Split: `{summary['split']}`",
        f"- Selected levels: {', '.join(f'`{name}`' for name in summary['selected_level_names'])}",
        f"- Coarsest level: `{summary['coarsest_level_name']}`",
        f"- Post-concat normalization: `{summary['post_concat_normalization']}`",
        f"- Mean `{CANONICAL_PRIMARY_METRIC}`: `{summary['mean_metrics'][CANONICAL_PRIMARY_METRIC]:.6f}`",
        f"- Mean `{CANONICAL_SECONDARY_METRIC}`: `{summary['mean_metrics'][CANONICAL_SECONDARY_METRIC]:.6f}`",
        f"- Final weight norms: `{json.dumps(summary['final_level_weight_norms'], sort_keys=True)}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_linear_probe_run_config(args, *, run_kind: str) -> dict[str, Any]:
    return {
        "command": getattr(args, "command", None),
        "run_kind": run_kind,
        "dataset_id": resolve_stage2_dataset_id(args),
        "dataset_source": str(getattr(args, "dataset_source", "rwtd")),
        "route": getattr(args, "route", None),
        "benchmark_root": getattr(args, "benchmark_root", None),
        "dataset_root": getattr(args, "dataset_root", None),
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "split": getattr(args, "split", None),
        "variant": getattr(args, "variant", None),
        "model_id": args.model_id,
        "device": args.device,
        "official_checkpoint_path": args.official_checkpoint_path,
        "train_limit": getattr(args, "train_limit", None),
        "num_train_samples": getattr(args, "num_train_samples", getattr(args, "train_limit", None)),
        "eval_limit": getattr(args, "eval_limit", None),
        "limit": getattr(args, "limit", None),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "post_concat_normalization": bool(args.post_concat_normalization),
        "save_visuals": bool(args.save_visuals),
        "seed": int(getattr(args, "seed", 0)),
        "linear_probe_settings": build_linear_probe_settings_from_args(args),
        "variant_specs": LINEAR_PROBE_VARIANT_SPECS,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_linear_probe_settings_from_args(args) -> dict[str, Any]:
    settings = dict(COARSE_VS_FINE_LINEAR_PROBE_SETTINGS)
    settings.update(
        {
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "post_concat_normalization": bool(args.post_concat_normalization),
        }
    )
    return settings


def build_linear_probe_experiment_terms_markdown(
    args,
    *,
    run_kind: str,
    selected_level_names: tuple[str, ...],
    raw_level_shapes: dict[str, tuple[int, int, int]],
    aligned_level_shapes: dict[str, tuple[int, int, int]],
    coarsest_level_name: str,
    coarsest_grid_size: tuple[int, int],
    materialization_mode: str,
    train_sample_count: int | None,
    eval_sample_count: int | None,
    checkpoint_path: Path | None = None,
) -> str:
    if not EXPERIMENT_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Missing experiment contract file: {EXPERIMENT_CONTRACT_PATH}")
    variant = str(getattr(args, "variant", "unknown"))
    variant_spec = LINEAR_PROBE_VARIANT_SPECS[variant]
    dataset_source = str(getattr(args, "dataset_source", "rwtd"))
    resolved_dataset_id = resolve_stage2_dataset_id(args)
    run_scope_bullets = [
        f"Command family: `{getattr(args, 'command', 'coarse-vs-fine-linear-probe')}`.",
        f"Run kind: `{run_kind}`.",
        f"Variant: `{variant}`.",
        f"Variant summary: {variant_spec['summary']}",
        f"Dataset: `{resolved_dataset_id}`.",
        f"Model: `{args.model_id}` on device request `{args.device}`.",
        f"Saved visuals: `{bool(getattr(args, 'save_visuals', True))}`.",
        f"Repository-wide metric contract: `{CANONICAL_EVALUATION_CONTRACT}` with primary metrics `{CANONICAL_PRIMARY_METRIC}` and `{CANONICAL_SECONDARY_METRIC}`.",
    ]
    if checkpoint_path is not None:
        run_scope_bullets.append(f"Linear-probe checkpoint consumed by this run: `{checkpoint_path}`.")
    if dataset_source == "architexture_binary":
        run_scope_bullets.append(f"ArchiTexture route: `{getattr(args, 'route', None)}`.")
        run_scope_bullets.append(f"Official split root: `{getattr(args, 'benchmark_root', None)}`.")
    if dataset_source == "cstd_binary":
        run_scope_bullets.append(f"CSTD official split root: `{getattr(args, 'dataset_root', None)}`.")

    split_bullets: list[str] = []
    if run_kind == "train":
        split_bullets.extend(
            [
                f"Training split: `{args.train_split}` with effective cap `{getattr(args, 'train_limit', None)}`; loaded `{train_sample_count}` sample(s).",
                f"Post-training evaluation split: `{args.eval_split}` with limit `{getattr(args, 'eval_limit', None)}`; loaded `{eval_sample_count}` sample(s).",
                (
                    f"Few-shot training setting: `--num-train-samples {getattr(args, 'num_train_samples', None)}`."
                    if getattr(args, "num_train_samples", None) is not None
                    else "Few-shot training setting: disabled."
                ),
                "Validation split: none. This command family uses one train split and one post-training evaluation split.",
            ]
        )
    else:
        split_bullets.extend(
            [
                f"Evaluation split: `{args.split}` with limit `{getattr(args, 'limit', None)}`; loaded `{eval_sample_count}` sample(s).",
                "Train/validation separation is inherited from the checkpoint-producing run.",
            ]
        )

    feature_bullets = [
        "Frozen SAM feature source: main `backbone_fpn` pyramid reused from the coarse-vs-fine Stage-2 extractor.",
        f"Selected levels for the active variant: {', '.join(f'`{name}`' for name in selected_level_names)}.",
        f"Alignment reference level: `{coarsest_level_name}` with aligned grid `{tuple(int(v) for v in coarsest_grid_size)}`.",
        "Per-level normalization is inherited from the shared frozen feature extractor.",
        ("Post-concat normalization: enabled." if bool(args.post_concat_normalization) else "Post-concat normalization: disabled."),
        f"Feature materialization mode for this run: `{materialization_mode}`.",
    ]
    feature_bullets.extend(f"Raw level shape {item}." for item in format_shape_mapping(raw_level_shapes))
    feature_bullets.extend(f"Aligned level shape {item}." for item in format_shape_mapping(aligned_level_shapes))

    architecture_bullets = [
        "The model is a strict linear probe: concatenate the selected aligned feature levels along channels and apply one learnable `1x1` convolution to predict the two texture classes.",
        "There is no learned decoder, no affinity objective, no k-means, no nonlinear output head, and no feature unfreezing.",
        f"Post-concat normalization flag: `{bool(args.post_concat_normalization)}`.",
    ]

    objective_bullets = [
        "Training target is direct per-pixel two-class supervision on the coarsest aligned grid.",
        "Ground-truth labels are derived from the same binary texture masks used elsewhere in the repo.",
        "Background / unlabeled coarsest-grid pixels are ignored in the cross-entropy loss.",
        "If only one texture class survives on the coarsest grid for a sample, the sample still contributes valid supervised pixels for the surviving class.",
        f"Optimizer: `AdamW(lr={float(args.learning_rate)}, wd={float(args.weight_decay)})` for `{int(args.num_epochs)}` epoch(s).",
    ]

    evaluation_bullets = [
        "Evaluation path: frozen SAM -> aligned selected levels -> linear `1x1` logits on coarsest grid -> argmax binary class map -> nearest-neighbor upsample to image size -> permutation-invariant binary assignment -> `eval_miou` and `eval_ari`.",
        "Unlike the Stage-2 gated-probe path, there is no clustering step in this experiment.",
    ]

    output_bullets = [
        f"Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl` under `{render_relative_path(resolve_linear_probe_output_root(args))}`.",
        "Per-sample artifacts: `sample_rows/<index>.json`, `masks/<index>.npz`, and `visuals/<index>.png` when `--save-visuals` is enabled.",
        "Training runs additionally write `checkpoint.pt`, `weight_norms.json`, and `train_history.csv`.",
    ]

    failure_bullets = [
        "Hard-fail on missing feature levels, inconsistent tensor ranks, inconsistent aligned grids, or NaN/Inf features/logits.",
        "Hard-fail when a sample has no valid texture-region supervision pixels at all on the coarsest grid.",
        "Hard-fail when the linear-probe prediction is not binary after argmax or when metric computation cannot assign a valid binary partition.",
    ]

    sections = [
        ExperimentTermsSection(
            title="Experiment Scope",
            paragraphs=(
                "This document describes only the few-shot linear-probe experiment that produced this results directory.",
            ),
            bullets=tuple(run_scope_bullets),
        ),
        ExperimentTermsSection(
            title="Scientific Idea",
            bullets=(
                "This run is a supervised few-shot baseline on top of the same frozen coarse-vs-fine SAM features used by the Stage-2 gated probe.",
                "The goal is to test how much direct linear separability the selected frozen SAM scales already expose for binary texture partitioning.",
            ),
        ),
        ExperimentTermsSection(title="Data And Split Separation", bullets=tuple(split_bullets)),
        ExperimentTermsSection(title="Feature Extraction And Alignment", bullets=tuple(feature_bullets)),
        ExperimentTermsSection(title="Linear Probe Architecture", bullets=tuple(architecture_bullets)),
        ExperimentTermsSection(title="Training Objective", bullets=tuple(objective_bullets)),
        ExperimentTermsSection(title="Evaluation Protocol", bullets=tuple(evaluation_bullets)),
        ExperimentTermsSection(title="Outputs In This Results Directory", bullets=tuple(output_bullets)),
        ExperimentTermsSection(title="Failure Semantics", bullets=tuple(failure_bullets)),
    ]
    return render_experiment_terms_markdown(
        title="Coarse-vs-Fine SAM Scales Few-Shot Linear Probe Experiment Terms",
        summary_lines=(
            "This file is the run-local explanation for the few-shot linear-probe follow-up in the coarse-vs-fine SAM study.",
            "It records the exact dataset split, frozen feature path, linear architecture, training objective, evaluation semantics, and output files for this results directory.",
        ),
        sections=sections,
        related_paths=(
            render_relative_path(EXPERIMENT_CONTRACT_PATH),
            render_relative_path(Path("src/rwtd_sam3/models/sam3_coarse_vs_fine_linear_probe.py")),
            render_relative_path(Path("src/rwtd_sam3/eval/coarse_vs_fine_linear_probe.py")),
        ),
    )


def _build_linear_probe_module_from_sample(sample_bundle: CoarseVsFineScaleSample, *, args) -> Any:
    return LinearSegmentationProbeModule(
        level_input_dims={
            level_name: int(sample_bundle.aligned_level_shapes[level_name][0])
            for level_name in sample_bundle.selected_level_names
        },
        post_concat_normalization=bool(args.post_concat_normalization),
    ).module


def _build_linear_probe_module_from_checkpoint(checkpoint: dict[str, Any]) -> Any:
    return LinearSegmentationProbeModule(
        level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
        post_concat_normalization=bool(checkpoint["post_concat_normalization"]),
    ).module


def _save_linear_probe_result_bundle(
    *,
    output_dir: Path,
    sample_bundle: CoarseVsFineScaleSample,
    result: LinearProbeEvalBundle,
    save_visuals: bool,
) -> None:
    sample_index = int(sample_bundle.sample.index)
    (output_dir / "sample_rows").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sample_rows" / f"{sample_index}.json", result.row)
    np.savez_compressed(
        output_dir / "masks" / f"{sample_index}.npz",
        prediction_a=np.asarray(result.prediction_a, dtype=bool),
        prediction_b=np.asarray(result.prediction_b, dtype=bool),
        target_a=np.asarray(sample_bundle.sample.texture_a_mask, dtype=bool),
        target_b=np.asarray(sample_bundle.sample.texture_b_mask, dtype=bool),
        coarsest_grid_labels=np.asarray(sample_bundle.coarsest_grid_labels, dtype=np.int32),
    )
    if save_visuals:
        save_prediction_panel(
            output_path=output_dir / "visuals" / f"{sample_index}.png",
            sample=sample_bundle.sample,
            prediction_a=result.prediction_a,
            prediction_b=result.prediction_b,
            protocol=f"coarse_vs_fine_linear_probe:{result.row['variant']}",
            metric_summary=result.metric_summary,
        )
