"""Stage-2 coarse-vs-fine SAM-scale probe training and evaluation entrypoints.

This module owns a small learned follow-up to the Stage-1 coarse-vs-fine
experiment contract. It keeps SAM frozen, learns only a tiny interpretable
scale-gated probe on aligned multiscale SAM features, clusters the learned
embeddings with 2-way k-means, and scores the resulting binary partition with
exactly the same invariant mIoU / ARI semantics used elsewhere in the repo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from rwtd_sam3.data.architexture_binary import (
    get_architexture_binary_sample,
    load_architexture_binary_overview,
)
from rwtd_sam3.data.cstd_binary import get_cstd_binary_sample, load_cstd_binary_overview
from rwtd_sam3.data.rwtd import RwtdDecodedDataset, load_split_overview
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
from rwtd_sam3.models.sam3_coarse_vs_fine_scale_probe import (
    COARSE_VS_FINE_STAGE2_SETTINGS,
    STAGE2_VARIANT_SPECS,
    CoarseVsFineScaleSample,
    Sam3CoarseVsFineScaleFeatureExtractor,
    Sam3CoarseVsFineScaleProbeRuntimeError,
    ScaleGatedSamProbeModule,
    cluster_embedding_map_kmeans,
    compute_sampled_pairwise_affinity_loss,
    evaluate_probe_partition,
    materialize_feature_levels_on_device,
    resolve_probe_variant_levels,
)
from rwtd_sam3.utils.visualization import save_prediction_panel

LOGGER = logging.getLogger(__name__)
COARSE_VS_FINE_OUTPUT_ROOT = Path("outputs") / "rwtd_sam3_auto" / "coarse_vs_fine_sam_scales" / "stage2"
ARCHITEXTURE_COARSE_VS_FINE_OUTPUT_ROOT = Path("outputs") / "architexture_binary" / "coarse_vs_fine_sam_scales" / "stage2"
CSTD_COARSE_VS_FINE_OUTPUT_ROOT = Path("outputs") / "cstd_binary" / "coarse_vs_fine_sam_scales" / "stage2"
EXPERIMENT_CONTRACT_PATH = Path("experiments") / "coarse_vs_fine_sam_scales" / "experiment_contract.md"
FEATURE_MATERIALIZATION_MAX_BYTES = 1 << 30


@dataclass(frozen=True)
class ProbeTrainingArtifacts:
    checkpoint_path: Path
    initial_gate_logits: list[float]
    initial_gate_weights: dict[str, float]
    final_gate_logits: list[float]
    final_gate_weights: dict[str, float]
    training_history_rows: list[dict[str, Any]]
    level_input_dims: dict[str, int]


@dataclass(frozen=True)
class ProbeEvalBundle:
    row: dict[str, Any]
    metric_summary: str
    prediction_a: np.ndarray
    prediction_b: np.ndarray


@dataclass(frozen=True)
class FeatureSampleSource:
    extractor: Sam3CoarseVsFineScaleFeatureExtractor
    variant: str
    split: str
    dataset_id: str
    dataset_source: str
    dataset_route: str | None
    dataset_root: str | None
    samples: Any
    selected_level_names: tuple[str, ...]
    alignment_reference_level_names: tuple[str, ...]
    reference_sample_index: int
    reference_bundle: CoarseVsFineScaleSample
    cached_feature_bundles: tuple[CoarseVsFineScaleSample, ...] | None
    materialization_mode: str


@dataclass(frozen=True)
class LazyDecodedSampleSource:
    """Indexable lazy decoded-sample source for large Stage-2 splits."""

    num_samples: int
    get_sample_at_index: Any

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, index: int) -> Any:
        normalized_index = int(index)
        if normalized_index < 0 or normalized_index >= self.num_samples:
            raise IndexError(
                f"LazyDecodedSampleSource index {normalized_index} is out of range for {self.num_samples} samples."
            )
        return self.get_sample_at_index(normalized_index)


def determine_feature_materialization_mode(
    *,
    variant: str,
    sample_count: int,
    reference_bundle: CoarseVsFineScaleSample,
) -> str:
    """Return how one Stage-2 split should materialize prepared features."""

    del variant
    estimated_total_bytes = int(sample_count) * estimate_feature_bundle_nbytes(reference_bundle)
    if estimated_total_bytes > FEATURE_MATERIALIZATION_MAX_BYTES:
        return "streamed_reprepare"
    return "in_memory"


def iter_feature_bundles(source: FeatureSampleSource):
    """Yield prepared feature bundles for one Stage-2 split source."""

    if source.cached_feature_bundles is not None:
        yield from source.cached_feature_bundles
        return
    yield source.reference_bundle
    for sample_index in range(len(source.samples)):
        if sample_index == int(source.reference_sample_index):
            continue
        sample = source.samples[sample_index]
        try:
            bundle = source.extractor.prepare_sample(
                sample,
                selected_level_names=source.selected_level_names,
                alignment_reference_level_names=source.alignment_reference_level_names,
            )
        except Sam3CoarseVsFineScaleProbeRuntimeError as exc:
            if is_skippable_collapsed_gt_alignment_error(exc):
                LOGGER.warning(
                    "Skipping Stage-2 sample index=%d crop_name=%s split=%s because the coarsest-grid GT alignment collapsed one region.",
                    int(sample.index),
                    sample.crop_name,
                    getattr(sample, "split", source.split),
                )
                continue
            raise
        validate_feature_bundle_against_reference(bundle=bundle, reference_bundle=source.reference_bundle)
        yield bundle


def estimate_feature_bundle_nbytes(bundle: CoarseVsFineScaleSample) -> int:
    """Estimate the CPU memory footprint of one prepared Stage-2 feature bundle."""

    total_bytes = int(bundle.coarsest_grid_labels.nbytes)
    total_bytes += sum(int(feature_map.nbytes) for feature_map in bundle.aligned_feature_levels.values())
    return total_bytes


def validate_feature_bundle_against_reference(
    *,
    bundle: CoarseVsFineScaleSample,
    reference_bundle: CoarseVsFineScaleSample,
) -> None:
    """Validate one prepared Stage-2 bundle against the reference sample signature."""

    if bundle.raw_level_names != reference_bundle.raw_level_names:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Stage-2 discovered inconsistent SAM level-name sets across samples.",
            diagnostics={
                "reference_raw_names": reference_bundle.raw_level_names,
                "current_raw_names": bundle.raw_level_names,
                "crop_name": bundle.sample.crop_name,
            },
        )
    for level_name in reference_bundle.raw_level_names:
        reference_channels = int(reference_bundle.raw_level_shapes[level_name][0])
        current_channels = int(bundle.raw_level_shapes[level_name][0])
        if current_channels != reference_channels:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "Stage-2 discovered inconsistent SAM channel counts across samples.",
                diagnostics={
                    "level_name": level_name,
                    "reference_channels": reference_channels,
                    "current_channels": current_channels,
                    "crop_name": bundle.sample.crop_name,
                },
            )


def is_skippable_collapsed_gt_alignment_error(error: Sam3CoarseVsFineScaleProbeRuntimeError) -> bool:
    """Return whether one Stage-2 sample can be skipped with an explicit warning."""

    return "GT alignment collapsed one of the two texture regions on the coarsest feature grid." in str(error)


def resolve_stage2_dataset_id(args) -> str:
    dataset_source = str(getattr(args, "dataset_source", "rwtd"))
    if dataset_source == "architexture_binary":
        route = getattr(args, "route", None)
        if route is None:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "Stage-2 ArchiTexture runs require --route.",
            )
        return f"architexture:{route}"
    if dataset_source == "cstd_binary":
        return "cstd"
    return str(args.dataset_id)


def resolve_stage2_output_root(args) -> Path:
    dataset_source = str(getattr(args, "dataset_source", "rwtd"))
    if dataset_source == "architexture_binary":
        return ARCHITEXTURE_COARSE_VS_FINE_OUTPUT_ROOT
    if dataset_source == "cstd_binary":
        return CSTD_COARSE_VS_FINE_OUTPUT_ROOT
    return COARSE_VS_FINE_OUTPUT_ROOT


def run_coarse_vs_fine_sam_probe_train(args) -> dict[str, Any]:
    """Train one Stage-2 scale-gated probe and evaluate it on the configured eval split."""

    output_dir = prepare_coarse_vs_fine_output_dir(
        args.output_dir,
        args=args,
        variant=args.variant,
        run_kind="train",
        split=args.eval_split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_stage2_run_config(args, run_kind="train")
    write_json(output_dir / "config.json", config)

    extractor = Sam3CoarseVsFineScaleFeatureExtractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token,
        official_checkpoint_path=args.official_checkpoint_path,
        settings=build_stage2_settings_from_args(args),
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
        variant=args.variant,
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
        variant=args.variant,
    )
    train_reference_bundle = train_samples.reference_bundle
    write_text(
        output_dir / "experiment_terms.md",
        build_stage2_experiment_terms_markdown(
            args,
            run_kind="train",
            selected_level_names=train_reference_bundle.selected_level_names,
            raw_level_shapes=train_reference_bundle.raw_level_shapes,
            aligned_level_shapes=train_reference_bundle.aligned_level_shapes,
            coarsest_level_name=train_reference_bundle.coarsest_level_name,
            coarsest_grid_size=train_reference_bundle.coarsest_grid_size,
            materialization_mode=train_samples.materialization_mode,
            train_sample_count=len(train_samples.samples),
            eval_sample_count=len(eval_samples.samples),
        ),
    )

    probe_module, training_artifacts = train_scale_gated_probe(
        train_samples=train_samples,
        output_dir=output_dir,
        args=args,
    )
    return evaluate_scale_gated_probe(
        probe_module=probe_module,
        eval_samples=eval_samples,
        output_dir=output_dir,
        args=args,
        checkpoint_path=training_artifacts.checkpoint_path,
        checkpoint_metadata={
            "variant": args.variant,
            "selected_level_names": list(train_reference_bundle.selected_level_names),
            "coarsest_level_name": train_reference_bundle.coarsest_level_name,
            "projection_dim": int(args.projection_dim),
            "embedding_dim": int(args.embedding_dim),
            "learnable_gates": bool(STAGE2_VARIANT_SPECS[args.variant]["learnable_gates"]),
            "level_input_dims": training_artifacts.level_input_dims,
            "initial_gate_logits": training_artifacts.initial_gate_logits,
            "initial_gate_weights": training_artifacts.initial_gate_weights,
            "final_gate_logits": training_artifacts.final_gate_logits,
            "final_gate_weights": training_artifacts.final_gate_weights,
            "train_history_rows": training_artifacts.training_history_rows,
            "pairs_per_image": int(args.pairs_per_image),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "kmeans_metric": str(args.kmeans_metric),
            "kmeans_num_clusters": int(args.kmeans_num_clusters),
            "kmeans_max_iterations": int(args.kmeans_max_iterations),
            "kmeans_convergence_tolerance": float(args.kmeans_convergence_tolerance),
            "seed": int(getattr(args, "seed", 0)),
            "materialization_mode": train_samples.materialization_mode,
        },
        run_kind="train",
    )

def run_coarse_vs_fine_sam_probe_eval(args) -> dict[str, Any]:
    """Load one trained Stage-2 probe checkpoint and evaluate it on a split."""

    import torch

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint path does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_variant = str(checkpoint.get("variant"))
    setattr(args, "variant", checkpoint_variant)
    if checkpoint_variant not in STAGE2_VARIANT_SPECS:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Checkpoint stored unknown Stage-2 variant '{checkpoint_variant}'."
        )

    output_dir = prepare_coarse_vs_fine_output_dir(
        args.output_dir,
        args=args,
        variant=checkpoint_variant,
        run_kind="eval",
        split=args.split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_stage2_run_config(args, run_kind="eval")
    config["checkpoint_path"] = str(checkpoint_path)
    config["variant"] = checkpoint_variant
    write_json(output_dir / "config.json", config)

    extractor = Sam3CoarseVsFineScaleFeatureExtractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token,
        official_checkpoint_path=args.official_checkpoint_path,
        settings=build_stage2_settings_from_args(args),
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
        variant=checkpoint_variant,
        expected_selected_level_names=tuple(checkpoint["selected_level_names"]),
    )
    eval_reference_bundle = eval_samples.reference_bundle
    write_text(
        output_dir / "experiment_terms.md",
        build_stage2_experiment_terms_markdown(
            args,
            run_kind="eval",
            selected_level_names=eval_reference_bundle.selected_level_names,
            raw_level_shapes=eval_reference_bundle.raw_level_shapes,
            aligned_level_shapes=eval_reference_bundle.aligned_level_shapes,
            coarsest_level_name=eval_reference_bundle.coarsest_level_name,
            coarsest_grid_size=eval_reference_bundle.coarsest_grid_size,
            materialization_mode=eval_samples.materialization_mode,
            train_sample_count=None,
            eval_sample_count=len(eval_samples.samples),
            checkpoint_path=checkpoint_path,
        ),
    )
    probe_module = _build_probe_module_from_checkpoint(checkpoint)
    probe_module.load_state_dict(checkpoint["probe_state_dict"])
    return evaluate_scale_gated_probe(
        probe_module=probe_module,
        eval_samples=eval_samples,
        output_dir=output_dir,
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint,
        run_kind="eval",
    )

def train_scale_gated_probe(
    *,
    train_samples: FeatureSampleSource,
    output_dir: Path,
    args,
) -> tuple[Any, ProbeTrainingArtifacts]:
    """Optimize only the tiny Stage-2 probe parameters on cached frozen features."""

    import torch

    if not train_samples.samples:
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Stage-2 training received an empty train split.")
    reference_bundle = train_samples.reference_bundle
    probe_module = _build_probe_module_from_sample(reference_bundle, variant=args.variant, args=args)
    train_device = _resolve_torch_device(args.device)
    probe_module.to(train_device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in probe_module.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    initial_gate_logits = _tensor_to_float_list(probe_module.current_gate_logits())
    initial_gate_weights = _gate_weight_dict(probe_module.level_names, probe_module.current_gate_weights())
    history_rows: list[dict[str, Any]] = []

    LOGGER.info(
        "Stage-2 probe training start | variant=%s levels=%s coarsest=%s aligned_shapes=%s materialization_mode=%s d=%d d_emb=%d learned_gates=%s pairs_per_image=%d optimizer=AdamW lr=%.6f wd=%.6f kmeans=%s/%d metrics=%s,%s",
        args.variant,
        list(reference_bundle.selected_level_names),
        reference_bundle.coarsest_level_name,
        reference_bundle.aligned_level_shapes,
        train_samples.materialization_mode,
        int(args.projection_dim),
        int(args.embedding_dim),
        bool(STAGE2_VARIANT_SPECS[args.variant]["learnable_gates"]),
        int(args.pairs_per_image),
        float(args.learning_rate),
        float(args.weight_decay),
        str(args.kmeans_metric),
        int(args.kmeans_num_clusters),
        CANONICAL_PRIMARY_METRIC,
        CANONICAL_SECONDARY_METRIC,
    )

    for epoch in range(int(args.num_epochs)):
        probe_module.train()
        epoch_losses: list[float] = []
        epoch_same: list[float] = []
        epoch_diff: list[float] = []
        for sample_bundle in iter_feature_bundles(train_samples):
            feature_levels = materialize_feature_levels_on_device(
                sample_bundle.aligned_feature_levels,
                device=train_device,
            )
            labels = torch.as_tensor(sample_bundle.coarsest_grid_labels, dtype=torch.long, device=train_device)
            optimizer.zero_grad(set_to_none=True)
            output = probe_module(feature_levels)
            loss_result = compute_sampled_pairwise_affinity_loss(
                output.embedding,
                labels,
                pairs_per_image=int(args.pairs_per_image),
                affinity_temperature=float(args.affinity_temperature),
            )
            loss_result.loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss_result.loss.detach().cpu().item()))
            epoch_same.append(float(loss_result.mean_same_similarity))
            epoch_diff.append(float(loss_result.mean_different_similarity))
        if not epoch_losses:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "Stage-2 training produced no valid samples after coarsest-grid GT-alignment filtering.",
                diagnostics={
                    "variant": args.variant,
                    "split": train_samples.split,
                    "dataset_id": train_samples.dataset_id,
                },
            )
        history_rows.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(mean(epoch_losses)),
                "mean_same_similarity": float(mean(epoch_same)),
                "mean_different_similarity": float(mean(epoch_diff)),
            }
        )

    final_gate_logits = _tensor_to_float_list(probe_module.current_gate_logits())
    final_gate_weights = _gate_weight_dict(probe_module.level_names, probe_module.current_gate_weights())
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "variant": args.variant,
            "model_id": args.model_id,
            "selected_level_names": list(reference_bundle.selected_level_names),
            "coarsest_level_name": reference_bundle.coarsest_level_name,
            "projection_dim": int(args.projection_dim),
            "embedding_dim": int(args.embedding_dim),
            "learnable_gates": bool(STAGE2_VARIANT_SPECS[args.variant]["learnable_gates"]),
            "level_input_dims": {
                level_name: int(reference_bundle.aligned_level_shapes[level_name][0])
                for level_name in reference_bundle.selected_level_names
            },
            "probe_state_dict": probe_module.state_dict(),
            "seed": int(getattr(args, "seed", 0)),
            "pairs_per_image": int(args.pairs_per_image),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "affinity_temperature": float(args.affinity_temperature),
            "kmeans_metric": str(args.kmeans_metric),
            "kmeans_num_clusters": int(args.kmeans_num_clusters),
            "kmeans_max_iterations": int(args.kmeans_max_iterations),
            "kmeans_convergence_tolerance": float(args.kmeans_convergence_tolerance),
            "materialization_mode": train_samples.materialization_mode,
            "initial_gate_logits": initial_gate_logits,
            "initial_gate_weights": initial_gate_weights,
            "final_gate_logits": final_gate_logits,
            "final_gate_weights": final_gate_weights,
            "train_history_rows": history_rows,
        },
        checkpoint_path,
    )
    write_csv(output_dir / "train_history.csv", history_rows)
    write_json(
        output_dir / "gate_weights.json",
        {
            "initial_gate_logits": initial_gate_logits,
            "initial_gate_weights": initial_gate_weights,
            "final_gate_logits": final_gate_logits,
            "final_gate_weights": final_gate_weights,
        },
    )
    return probe_module, ProbeTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        initial_gate_logits=initial_gate_logits,
        initial_gate_weights=initial_gate_weights,
        final_gate_logits=final_gate_logits,
        final_gate_weights=final_gate_weights,
        training_history_rows=history_rows,
        level_input_dims={
            level_name: int(reference_bundle.aligned_level_shapes[level_name][0])
            for level_name in reference_bundle.selected_level_names
        },
    )

def evaluate_scale_gated_probe(
    *,
    probe_module: Any,
    eval_samples: FeatureSampleSource,
    output_dir: Path,
    args,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    run_kind: str,
) -> dict[str, Any]:
    """Evaluate a trained Stage-2 probe using the Stage-1 clustering semantics."""

    import torch

    probe_module.eval()
    eval_device = _resolve_torch_device(args.device)
    probe_module.to(eval_device)
    visual_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for sample_bundle in iter_feature_bundles(eval_samples):
        with torch.no_grad():
            feature_levels = materialize_feature_levels_on_device(
                sample_bundle.aligned_feature_levels,
                device=eval_device,
            )
            output = probe_module(feature_levels)
            label_map = cluster_embedding_map_kmeans(
                output.embedding,
                num_clusters=int(args.kmeans_num_clusters),
                metric=str(args.kmeans_metric),
                max_iterations=int(args.kmeans_max_iterations),
                tolerance=float(args.kmeans_convergence_tolerance),
            )
        partition = evaluate_probe_partition(sample=sample_bundle.sample, coarsest_grid_label_map=label_map)
        row = build_probe_eval_row(
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
            f"Gates={row['final_gate_weights_compact']}"
        )
        result = ProbeEvalBundle(
            row=row,
            metric_summary=metric_summary,
            prediction_a=partition.prediction_a,
            prediction_b=partition.prediction_b,
        )
        _save_result_bundle(
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
                    protocol=f"coarse_vs_fine_stage2:{checkpoint_metadata['variant']}",
                    visual_path=Path("visuals") / f"{sample_bundle.sample.index}.png",
                )
            )

    if not rows:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Stage-2 evaluation produced no valid samples after coarsest-grid GT-alignment filtering.",
            diagnostics={
                "variant": str(checkpoint_metadata["variant"]),
                "split": eval_samples.split,
                "dataset_id": eval_samples.dataset_id,
            },
        )

    write_csv(output_dir / "per_sample_metrics.csv", rows)
    write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
    write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)

    summary = build_probe_summary(
        rows=rows,
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        run_kind=run_kind,
    )
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", [flatten_probe_summary(summary)])
    write_text(output_dir / "summary.md", build_probe_summary_markdown(summary))
    return summary

def build_probe_eval_row(
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
    gate_logits = _tensor_to_float_list(probe_module.current_gate_logits())
    gate_weights = _gate_weight_dict(probe_module.level_names, probe_module.current_gate_weights())
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
        "probe_projection_dim": int(args.projection_dim),
        "probe_embedding_dim": int(args.embedding_dim),
        "learnable_gates": bool(STAGE2_VARIANT_SPECS[variant]["learnable_gates"]),
        "gate_logits_json": json.dumps(gate_logits),
        "gate_weights_json": json.dumps(gate_weights, sort_keys=True),
        "final_gate_weights_compact": ",".join(f"{name}={weight:.3f}" for name, weight in gate_weights.items()),
        "kmeans_backend": "deterministic_kmeans",
        "kmeans_metric": str(args.kmeans_metric),
        "kmeans_num_clusters": int(args.kmeans_num_clusters),
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


def build_probe_summary(
    *,
    rows: list[dict[str, Any]],
    args,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    run_kind: str,
) -> dict[str, Any]:
    if not rows:
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Stage-2 evaluation produced no sample rows.")
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
        "benchmark_root": getattr(args, "benchmark_root", None),
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
        "projection_dim": int(checkpoint_metadata["projection_dim"]),
        "embedding_dim": int(checkpoint_metadata["embedding_dim"]),
        "learnable_gates": bool(checkpoint_metadata["learnable_gates"]),
        "pair_samples_per_image": int(checkpoint_metadata["pairs_per_image"]),
        "optimizer": "AdamW",
        "learning_rate": float(checkpoint_metadata["learning_rate"]),
        "weight_decay": float(checkpoint_metadata["weight_decay"]),
        "num_epochs": int(checkpoint_metadata["num_epochs"]),
        "kmeans_backend": "deterministic_kmeans",
        "kmeans_metric": str(checkpoint_metadata["kmeans_metric"]),
        "kmeans_num_clusters": int(checkpoint_metadata["kmeans_num_clusters"]),
        "kmeans_max_iterations": int(checkpoint_metadata["kmeans_max_iterations"]),
        "kmeans_convergence_tolerance": float(checkpoint_metadata["kmeans_convergence_tolerance"]),
        "seed": int(checkpoint_metadata["seed"]),
        "initial_gate_logits": list(checkpoint_metadata["initial_gate_logits"]),
        "initial_gate_weights": dict(checkpoint_metadata["initial_gate_weights"]),
        "final_gate_logits": list(checkpoint_metadata["final_gate_logits"]),
        "final_gate_weights": dict(checkpoint_metadata["final_gate_weights"]),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_probe_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Coarse-vs-Fine SAM Scales Stage 2 Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Dataset source: `{summary.get('dataset_source', 'rwtd')}`",
        f"- Split: `{summary['split']}`",
        f"- Selected levels: {', '.join(f'`{name}`' for name in summary['selected_level_names'])}",
        f"- Coarsest level: `{summary['coarsest_level_name']}`",
        f"- Probe dims: `d={summary['projection_dim']}`, `d_emb={summary['embedding_dim']}`",
        f"- Learned gates: `{summary['learnable_gates']}`",
        f"- Mean `{CANONICAL_PRIMARY_METRIC}`: `{summary['mean_metrics'][CANONICAL_PRIMARY_METRIC]:.6f}`",
        f"- Mean `{CANONICAL_SECONDARY_METRIC}`: `{summary['mean_metrics'][CANONICAL_SECONDARY_METRIC]:.6f}`",
        f"- Final gate weights: `{json.dumps(summary['final_gate_weights'], sort_keys=True)}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def flatten_probe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": summary["variant"],
        "split": summary["split"],
        "num_evaluated_samples": summary["num_evaluated_samples"],
        CANONICAL_PRIMARY_METRIC: summary["mean_metrics"][CANONICAL_PRIMARY_METRIC],
        CANONICAL_SECONDARY_METRIC: summary["mean_metrics"][CANONICAL_SECONDARY_METRIC],
        "coarsest_level_name": summary["coarsest_level_name"],
        "selected_level_names_json": json.dumps(summary["selected_level_names"]),
        "learnable_gates": summary["learnable_gates"],
        "final_gate_weights_json": json.dumps(summary["final_gate_weights"], sort_keys=True),
        "checkpoint_path": summary["checkpoint_path"],
    }


def build_stage2_run_config(args, *, run_kind: str) -> dict[str, Any]:
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
        "projection_dim": int(args.projection_dim),
        "embedding_dim": int(args.embedding_dim),
        "pairs_per_image": int(args.pairs_per_image),
        "affinity_temperature": float(args.affinity_temperature),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "kmeans_metric": str(args.kmeans_metric),
        "kmeans_num_clusters": int(args.kmeans_num_clusters),
        "kmeans_max_iterations": int(args.kmeans_max_iterations),
        "kmeans_convergence_tolerance": float(args.kmeans_convergence_tolerance),
        "save_visuals": args.save_visuals,
        "seed": int(getattr(args, "seed", 0)),
        "stage2_settings": build_stage2_settings_from_args(args),
        "variant_specs": STAGE2_VARIANT_SPECS,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_stage2_settings_from_args(args) -> dict[str, Any]:
    settings = dict(COARSE_VS_FINE_STAGE2_SETTINGS)
    settings.update(
        {
            "projection_dim": int(args.projection_dim),
            "embedding_dim": int(args.embedding_dim),
            "pair_samples_per_image": int(args.pairs_per_image),
            "affinity_temperature": float(args.affinity_temperature),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "kmeans_metric": str(args.kmeans_metric),
            "kmeans_num_clusters": int(args.kmeans_num_clusters),
            "kmeans_max_iterations": int(args.kmeans_max_iterations),
            "kmeans_convergence_tolerance": float(args.kmeans_convergence_tolerance),
        }
    )
    return settings


def build_stage2_experiment_terms_markdown(
    args,
    *,
    run_kind: str,
    selected_level_names: tuple[str, ...],
    raw_level_shapes: dict[str, tuple[int, int, int]],
    aligned_level_shapes: dict[str, tuple[int, int, int]],
    coarsest_level_name: str,
    coarsest_grid_size: tuple[int, int],
    materialization_mode: str = "in_memory",
    train_sample_count: int | None = None,
    eval_sample_count: int | None = None,
    checkpoint_path: Path | None = None,
) -> str:
    """Build a detailed Stage-2 experiment-specific ``experiment_terms.md`` document."""

    if not EXPERIMENT_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Missing experiment contract file: {EXPERIMENT_CONTRACT_PATH}")

    variant = str(getattr(args, "variant", "unknown"))
    variant_spec = STAGE2_VARIANT_SPECS[variant]
    dataset_source = str(getattr(args, "dataset_source", "rwtd"))
    resolved_dataset_id = resolve_stage2_dataset_id(args)
    output_root = resolve_stage2_output_root(args)
    selected_level_display = ", ".join(f"`{level}`" for level in selected_level_names)
    all_level_display = ", ".join(f"`{level}`" for level in raw_level_shapes)
    run_scope_bullets = [
        f"Command family: `{getattr(args, 'command', 'coarse-vs-fine-stage2')}`.",
        f"Run kind: `{run_kind}`.",
        f"Variant: `{variant}`.",
        f"Variant summary: {variant_spec['summary']}",
        f"Dataset: `{resolved_dataset_id}`.",
        f"Model: `{args.model_id}` on device request `{args.device}`.",
        f"Official checkpoint override for SAM feature extraction: `{args.official_checkpoint_path}`.",
        f"Saved visuals: `{bool(getattr(args, 'save_visuals', True))}`.",
        f"Repository-wide metric contract: `{CANONICAL_EVALUATION_CONTRACT}` with primary metrics `{CANONICAL_PRIMARY_METRIC}` and `{CANONICAL_SECONDARY_METRIC}`.",
    ]
    if checkpoint_path is not None:
        run_scope_bullets.append(f"Probe checkpoint consumed by this run: `{checkpoint_path}`.")
    if dataset_source == "architexture_binary":
        run_scope_bullets.append(f"ArchiTexture route: `{getattr(args, 'route', None)}`.")
        run_scope_bullets.append(f"Official split root: `{getattr(args, 'benchmark_root', None)}`.")
    if dataset_source == "cstd_binary":
        run_scope_bullets.append(f"CSTD official split root: `{getattr(args, 'dataset_root', None)}`.")

    split_bullets: list[str] = []
    effective_train_limit = getattr(args, "train_limit", None)
    declared_num_train_samples = getattr(args, "num_train_samples", effective_train_limit)
    if run_kind == "train":
        split_bullets.extend(
            [
                f"Training split: `{args.train_split}` with effective cap `{effective_train_limit}`; loaded `{train_sample_count}` sample(s).",
                f"Post-training evaluation split: `{args.eval_split}` with limit `{getattr(args, 'eval_limit', None)}`; loaded `{eval_sample_count}` sample(s).",
                (f"Few-shot training setting: `--num-train-samples {declared_num_train_samples}`. Only that many samples are loaded from the requested training split before optimization." if declared_num_train_samples is not None else "Few-shot training setting: disabled. The full resolved training split is used."),
                "Validation split: none. This command family uses one train split and one post-training evaluation split instead of a separate validation loop.",
                (
                    "Official split contract: Stage-2 ArchiTexture runs require a split-aware local root and explicitly reject the flat benchmark-only layout."
                    if dataset_source == "architexture_binary"
                    else "Official split contract: Stage-2 CSTD runs require a split-aware local root and explicitly reject the flat benchmark-only layout."
                    if dataset_source == "cstd_binary"
                    else "Dataset source: RWTD Hugging Face split loader."
                ),
            ]
        )
    else:
        split_bullets.extend(
            [
                f"Evaluation split: `{args.split}` with limit `{getattr(args, 'limit', None)}`; loaded `{eval_sample_count}` sample(s).",
                "Train/validation separation is inherited from the checkpoint-producing run. This eval command updates no weights and performs no additional fitting.",
                "Validation split: none in this command family; checkpoint evaluation is done directly on the requested split.",
                (
                    "Official split contract: Stage-2 ArchiTexture runs require a split-aware local root and explicitly reject the flat benchmark-only layout."
                    if dataset_source == "architexture_binary"
                    else "Official split contract: Stage-2 CSTD runs require a split-aware local root and explicitly reject the flat benchmark-only layout."
                    if dataset_source == "cstd_binary"
                    else "Dataset source: RWTD Hugging Face split loader."
                ),
            ]
        )

    feature_bullets = [
        f"Frozen SAM feature source: `{COARSE_VS_FINE_STAGE2_SETTINGS['feature_source']}` from the main image pyramid (`backbone_fpn`).",
        f"All discovered main pyramid levels for this run: {all_level_display}.",
        f"Selected levels for the active variant: {selected_level_display}.",
        f"Alignment reference level: `{coarsest_level_name}` with aligned grid `{tuple(int(v) for v in coarsest_grid_size)}`.",
        f"Feature alignment policy: resize every selected level to the coarsest selected grid with `{COARSE_VS_FINE_STAGE2_SETTINGS['feature_alignment_mode']}` interpolation before learning or clustering.",
        f"Per-level normalization: `{COARSE_VS_FINE_STAGE2_SETTINGS['per_level_normalization']}` across channels at every spatial location.",
        f"Post-fusion normalization: `{COARSE_VS_FINE_STAGE2_SETTINGS['post_fusion_normalization']}` on the final embedding map.",
        "Clustering never sees image-resolution features. Only the final binary label map is upsampled back to image size for permutation-invariant evaluation and visualization.",
    ]
    feature_bullets.extend(f"Raw level shape {item}." for item in format_shape_mapping(raw_level_shapes))
    feature_bullets.extend(f"Aligned level shape {item}." for item in format_shape_mapping(aligned_level_shapes))
    if len(selected_level_names) == 1:
        feature_bullets.append(
            "Single-scale comparison note: the selected level keeps its own native SAM grid. `fpn_1_only` therefore clusters on the native `144x144` grid, and `fpn_0_only` clusters on the native `288x288` grid."
        )
    feature_bullets.append(f"Feature materialization mode for this run: `{materialization_mode}`.")

    architecture_paragraphs = [
        "The Stage-2 model is a tiny interpretable probe on top of frozen SAM features. For each selected level `l`, the aligned normalized feature map `F_l` is projected with one learnable `1x1` convolution `P_l: C_l -> d`. Global scalar gate logits `g_l` are converted to scale weights `alpha = softmax(g)`, and the fused representation is `Z = sum_l alpha_l * P_l(F_l)`. A tiny output head maps `Z` from `d` to `d_emb` when needed, and the final per-pixel embedding is L2-normalized before clustering.",
        "The probe stays deliberately low-capacity: no decoder, no UNet, no transformer blocks, no image-resolution feature fusion, and no direct supervised segmentation head. The interpretation target is the learned scale weights themselves.",
    ]
    architecture_bullets = [
        f"Projection dim `d`: `{int(args.projection_dim)}`.",
        f"Embedding dim `d_emb`: `{int(args.embedding_dim)}`.",
        f"Learnable gates: `{bool(variant_spec['learnable_gates'])}`.",
        (
            "Gate behavior: one global learnable logit per selected level, shared across the whole dataset."
            if bool(variant_spec['learnable_gates'])
            else "Gate behavior: fixed uniform weights over the selected levels; logits are not trainable."
        ),
    ]

    objective_bullets = [
        "SAM stays frozen. Only the tiny scale-gated probe parameters are trainable.",
        "Ground-truth binary partitions are downsampled to the coarsest aligned grid with nearest-neighbor interpolation.",
        "Collapsed coarsest-grid GT is allowed. If nearest-neighbor alignment leaves only one texture region on the coarsest grid, the sample still contributes same-region affinity supervision from the surviving texture region.",
        f"Affinity supervision samples `{int(args.pairs_per_image)}` same/different pixel pairs per image on the coarsest grid.",
        f"Affinity similarity uses temperature-scaled cosine logits with temperature `{float(args.affinity_temperature)}` and a BCE-style objective.",
        f"Optimizer: `AdamW(lr={float(args.learning_rate)}, wd={float(args.weight_decay)})` for `{int(args.num_epochs)}` epoch(s).",
        f"Random seed: `{int(getattr(args, 'seed', 0))}`.",
    ]

    evaluation_bullets = [
        f"Evaluation path: frozen SAM -> aligned selected levels -> probe embedding -> deterministic k-means with `k={int(args.kmeans_num_clusters)}` -> nearest-neighbor upsample to image size -> permutation-invariant binary assignment -> `{CANONICAL_PRIMARY_METRIC}` and `{CANONICAL_SECONDARY_METRIC}`.",
        f"Clustering backend: deterministic k-means with metric `{args.kmeans_metric}`, max iterations `{int(args.kmeans_max_iterations)}`, tolerance `{float(args.kmeans_convergence_tolerance)}`.",
        "Probe outputs are never thresholded directly as the primary evaluation path. The main result always comes from clustering the learned embedding.",
    ]

    output_bullets = [
        f"Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl` under `{render_relative_path(output_root)}`.",
        "Per-sample artifacts: `sample_rows/<index>.json`, `masks/<index>.npz`, and `visuals/<index>.png` when `--save-visuals` is enabled.",
        "Mask bundles store `prediction_a`, `prediction_b`, `target_a`, `target_b`, and the coarsest-grid GT labels used for the affinity objective.",
    ]
    if run_kind == "train":
        output_bullets.append("Training runs additionally write `checkpoint.pt`, `gate_weights.json`, and `train_history.csv`.")
    else:
        output_bullets.append("Evaluation runs require `--checkpoint-path` and reuse the checkpointed probe weights without modification.")

    failure_bullets = [
        "Hard-fail on a missing experiment contract file, missing Stage-2 checkpoint path, or an unknown Stage-2 variant.",
        "Hard-fail on missing feature levels, inconsistent tensor ranks, inconsistent per-level channel counts across samples, or invalid coarsest-grid sizes.",
        "Hard-fail on NaN/Inf features, embeddings, similarities, or clustering inputs.",
        "Hard-fail when pair sampling yields no valid texture-region supervision pixels at all on the coarsest grid.",
        "Hard-fail when k-means does not return a binary label map or when metric computation cannot assign a valid binary partition.",
        "Hard-fail on a missing ArchiTexture official split layout when `--dataset-source architexture_binary` is requested.",
    ]

    runtime_bullets = [
        f"Variant policy: `{variant_spec['selected_level_policy']}`.",
        f"Projection dim argument: `{int(args.projection_dim)}`.",
        f"Embedding dim argument: `{int(args.embedding_dim)}`.",
        f"Pair samples per image argument: `{int(args.pairs_per_image)}`.",
        f"Affinity temperature argument: `{float(args.affinity_temperature)}`.",
        f"K-means arguments: metric=`{args.kmeans_metric}`, `k={int(args.kmeans_num_clusters)}`, `max_iterations={int(args.kmeans_max_iterations)}`, `tolerance={float(args.kmeans_convergence_tolerance)}`.",
        f"Feature materialization mode: `{materialization_mode}`.",
    ]

    sections = [
        ExperimentTermsSection(
            title="Experiment Scope",
            paragraphs=(
                "This document describes only the experiment that produced this results directory. It is intentionally experiment-specific and should be sufficient for a reviewer to understand the run without reading unrelated repository docs.",
            ),
            bullets=run_scope_bullets,
        ),
        ExperimentTermsSection(
            title="Hypothesis",
            bullets=(
                "Coarse SAM feature scales should carry most of the useful signal for binary texture-region partitioning.",
                "A tiny learned multi-scale adapter should therefore place most useful mass on coarse scales or show only weak gains from finer scales.",
                "Finer scales may still help local boundary placement, but they are not expected to dominate global partition quality on the selected binary texture benchmark.",
            ),
        ),
        ExperimentTermsSection(
            title="Scientific Idea",
            paragraphs=(
                "Stage 1 compared non-learned clustering baselines over frozen SAM scales. Stage 2 keeps the same frozen feature source and the same clustering-based evaluation semantics, but adds the smallest interpretable learned component: per-level `1x1` projections and one global gate per selected scale.",
                "The scientific question is not whether a large supervised decoder can solve the selected binary texture benchmark. The question is whether a tiny learned probe, when forced to work through clustering on frozen features, still prefers coarse scales.",
            ),
        ),
        ExperimentTermsSection(title="Data And Split Separation", bullets=tuple(split_bullets)),
        ExperimentTermsSection(title="Feature Extraction And Alignment", bullets=tuple(feature_bullets)),
        ExperimentTermsSection(
            title="Probe Architecture",
            paragraphs=tuple(architecture_paragraphs),
            bullets=tuple(architecture_bullets),
        ),
        ExperimentTermsSection(title="Training Objective", bullets=tuple(objective_bullets)),
        ExperimentTermsSection(title="Evaluation Protocol", bullets=tuple(evaluation_bullets)),
        ExperimentTermsSection(title="Runtime Settings", bullets=tuple(runtime_bullets)),
        ExperimentTermsSection(title="Outputs In This Results Directory", bullets=tuple(output_bullets)),
        ExperimentTermsSection(title="Failure Semantics", bullets=tuple(failure_bullets)),
    ]

    return render_experiment_terms_markdown(
        title="Coarse-vs-Fine SAM Scale Study: Stage 2 Experiment Terms",
        summary_lines=(
            "This file is the run-local explanation for the Stage-2 coarse-vs-fine SAM study.",
            "It records the exact hypothesis, method, split separation, architecture, evaluation path, outputs, and hard-failure rules for this specific results directory.",
        ),
        sections=sections,
        related_paths=(
            render_relative_path(EXPERIMENT_CONTRACT_PATH),
            render_relative_path(Path("src/rwtd_sam3/models/sam3_coarse_vs_fine_scale_probe.py")),
            render_relative_path(Path("src/rwtd_sam3/eval/coarse_vs_fine_sam_scales.py")),
            render_relative_path(Path("src/rwtd_sam3/eval/experiment_terms.py")),
        ),
    )


def prepare_coarse_vs_fine_output_dir(
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
        return resolve_stage2_output_root(args) / suffix
    return Path(output_dir)


def _load_stage2_raw_samples(
    *,
    dataset_source: str,
    dataset_id: str | None,
    cache_dir: str | None,
    route: str | None,
    benchmark_root: str | None,
    dataset_root: str | None,
    split: str,
    limit: int | None,
) -> tuple[str, Any]:
    if dataset_source == "rwtd":
        if dataset_id is None:
            raise Sam3CoarseVsFineScaleProbeRuntimeError("RWTD Stage-2 runs require --dataset-id.")
        overview = load_split_overview(split=split, dataset_id=dataset_id, cache_dir=cache_dir)
        requested_limit = overview.num_examples if limit is None else min(int(limit), int(overview.num_examples))
        samples = RwtdDecodedDataset(
            split=split,
            dataset_id=dataset_id,
            cache_dir=cache_dir,
            limit=requested_limit,
            length=requested_limit,
        )
        return str(dataset_id), samples
    if dataset_source == "architexture_binary":
        if route is None:
            raise Sam3CoarseVsFineScaleProbeRuntimeError("ArchiTexture Stage-2 runs require --route.")
        if benchmark_root is None:
            raise Sam3CoarseVsFineScaleProbeRuntimeError("ArchiTexture Stage-2 runs require --benchmark-root.")
        overview = load_architexture_binary_overview(
            route=route,
            benchmark_root=benchmark_root,
            split=split,
            require_official_split=True,
        )
        requested_limit = overview.num_examples if limit is None else min(int(limit), int(overview.num_examples))
        samples = LazyDecodedSampleSource(
            num_samples=requested_limit,
            get_sample_at_index=lambda index: get_architexture_binary_sample(
                benchmark_root=benchmark_root,
                route=route,
                index=int(index),
                split=split,
                require_official_split=True,
            ),
        )
        return f"architexture:{route}", samples
    if dataset_source == "cstd_binary":
        if dataset_root is None:
            raise Sam3CoarseVsFineScaleProbeRuntimeError("CSTD Stage-2 runs require --dataset-root.")
        overview = load_cstd_binary_overview(
            dataset_root=dataset_root,
            split=split,
            require_official_split=True,
        )
        requested_limit = overview.num_examples if limit is None else min(int(limit), int(overview.num_examples))
        samples = LazyDecodedSampleSource(
            num_samples=requested_limit,
            get_sample_at_index=lambda index: get_cstd_binary_sample(
                dataset_root=dataset_root,
                index=int(index),
                split=split,
                require_official_split=True,
            ),
        )
        return "cstd", samples
    raise Sam3CoarseVsFineScaleProbeRuntimeError(
        f"Unsupported Stage-2 dataset source '{dataset_source}'. Expected 'rwtd', 'architexture_binary', or 'cstd_binary'."
    )


def _load_feature_samples(
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
    expected_selected_level_names: tuple[str, ...] | None = None,
) -> FeatureSampleSource:
    resolved_dataset_id, samples = _load_stage2_raw_samples(
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
    available_level_names = tuple(first_pyramid.keys())
    selected_level_names = expected_selected_level_names or resolve_probe_variant_levels(
        variant=variant,
        available_level_names=available_level_names,
    )
    alignment_reference_level_names = selected_level_names
    reference_sample_index: int | None = None
    reference_bundle: CoarseVsFineScaleSample | None = None
    for sample_index in range(len(samples)):
        sample = samples[sample_index]
        try:
            reference_bundle = extractor.prepare_sample(
                sample,
                selected_level_names=selected_level_names,
                alignment_reference_level_names=alignment_reference_level_names,
            )
        except Sam3CoarseVsFineScaleProbeRuntimeError as exc:
            if is_skippable_collapsed_gt_alignment_error(exc):
                LOGGER.warning(
                    "Skipping Stage-2 reference-candidate sample index=%d crop_name=%s split=%s because the coarsest-grid GT alignment collapsed one region.",
                    int(sample.index),
                    sample.crop_name,
                    getattr(sample, "split", split),
                )
                continue
            raise
        reference_sample_index = sample_index
        break
    if reference_bundle is None or reference_sample_index is None:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "All Stage-2 samples collapsed one texture region on the coarsest feature grid; no valid reference sample remained.",
            diagnostics={
                "dataset_source": dataset_source,
                "dataset_id": resolved_dataset_id,
                "split": split,
                "requested_limit": limit,
            },
        )
    materialization_mode = determine_feature_materialization_mode(
        variant=variant,
        sample_count=len(samples),
        reference_bundle=reference_bundle,
    )
    estimated_bundle_bytes = estimate_feature_bundle_nbytes(reference_bundle)
    estimated_total_bytes = int(len(samples)) * estimated_bundle_bytes
    cached_feature_bundles: list[CoarseVsFineScaleSample] | None = (
        [reference_bundle] if materialization_mode == "in_memory" else None
    )
    if cached_feature_bundles is not None:
        for sample_index in range(len(samples)):
            if sample_index == reference_sample_index:
                continue
            sample = samples[sample_index]
            try:
                bundle = extractor.prepare_sample(
                    sample,
                    selected_level_names=selected_level_names,
                    alignment_reference_level_names=alignment_reference_level_names,
                )
            except Sam3CoarseVsFineScaleProbeRuntimeError as exc:
                if is_skippable_collapsed_gt_alignment_error(exc):
                    LOGGER.warning(
                        "Skipping Stage-2 cached sample index=%d crop_name=%s split=%s because the coarsest-grid GT alignment collapsed one region.",
                        int(sample.index),
                        sample.crop_name,
                        getattr(sample, "split", split),
                    )
                    continue
                raise
            validate_feature_bundle_against_reference(bundle=bundle, reference_bundle=reference_bundle)
            cached_feature_bundles.append(bundle)
    LOGGER.info(
        "Stage-2 prepared %d feature samples for split=%s | selected_levels=%s | coarsest=%s | materialization_mode=%s | estimated_bundle_mb=%.2f | estimated_total_gb=%.2f | first_raw_shapes=%s | first_aligned_shapes=%s",
        len(samples),
        split,
        list(reference_bundle.selected_level_names),
        reference_bundle.coarsest_level_name,
        materialization_mode,
        float(estimated_bundle_bytes / (1024 * 1024)),
        float(estimated_total_bytes / (1024 * 1024 * 1024)),
        reference_bundle.raw_level_shapes,
        reference_bundle.aligned_level_shapes,
    )
    return FeatureSampleSource(
        extractor=extractor,
        variant=variant,
        split=split,
        dataset_id=resolved_dataset_id,
        dataset_source=dataset_source,
        dataset_route=route,
        dataset_root=benchmark_root if dataset_source == "architexture_binary" else dataset_root,
        samples=samples,
        selected_level_names=selected_level_names,
        alignment_reference_level_names=alignment_reference_level_names,
        reference_sample_index=reference_sample_index,
        reference_bundle=reference_bundle,
        cached_feature_bundles=tuple(cached_feature_bundles) if cached_feature_bundles is not None else None,
        materialization_mode=materialization_mode,
    )


def _build_probe_module_from_sample(sample_bundle: CoarseVsFineScaleSample, *, variant: str, args) -> Any:
    return ScaleGatedSamProbeModule(
        level_input_dims={
            level_name: int(sample_bundle.aligned_level_shapes[level_name][0])
            for level_name in sample_bundle.selected_level_names
        },
        projection_dim=int(args.projection_dim),
        embedding_dim=int(args.embedding_dim),
        learnable_gates=bool(STAGE2_VARIANT_SPECS[variant]["learnable_gates"]),
    ).module


def _build_probe_module_from_checkpoint(checkpoint: dict[str, Any]) -> Any:
    return ScaleGatedSamProbeModule(
        level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
        projection_dim=int(checkpoint["projection_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        learnable_gates=bool(checkpoint["learnable_gates"]),
    ).module


def _resolve_torch_device(device_request: str) -> Any:
    import torch

    if device_request == "cuda":
        if not torch.cuda.is_available():
            raise Sam3CoarseVsFineScaleProbeRuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_request == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_float_list(tensor: Any) -> list[float]:
    import torch

    if torch.is_tensor(tensor):
        return [float(value) for value in tensor.detach().cpu().tolist()]
    return [float(value) for value in tensor]


def _gate_weight_dict(level_names: tuple[str, ...], gate_weights: Any) -> dict[str, float]:
    weights = _tensor_to_float_list(gate_weights)
    return {level_name: float(weights[index]) for index, level_name in enumerate(level_names)}


def _save_result_bundle(
    *,
    output_dir: Path,
    sample_bundle: CoarseVsFineScaleSample,
    result: ProbeEvalBundle,
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
            protocol=f"coarse_vs_fine_stage2:{result.row['variant']}",
            metric_summary=result.metric_summary,
        )
