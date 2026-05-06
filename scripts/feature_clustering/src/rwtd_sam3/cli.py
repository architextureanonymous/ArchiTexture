"""Command-line interface for RWTD and related benchmark evaluations.

This module defines the argparse surface exposed by ``main.py`` and the
installed ``rwtd-sam3`` console script. It is the single public entrypoint for
dataset inspection, prompt-conditioned SAM 3 evaluation, SAM-2 baselines,
SAM-3 automatic-mask experiments, the ArchiTexture route adapter, and the
local DeTexture ADE20K / CSTD / GlaS binary adapters.

Primary entrypoints:
- ``build_parser()``: build the root parser and every supported subcommand.
- ``main()``: parse arguments, validate shared constraints, and dispatch into
  the evaluation modules under ``rwtd_sam3.eval``.

Inputs are CLI flags and optional local dataset roots. Outputs are console logs
and run directories written by the downstream evaluation entrypoints. Runtime
dependencies depend on the selected command family and are documented in the
root README.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from rwtd_sam3.data.rwtd import DEFAULT_DATASET_ID, DEFAULT_EVAL_SPLIT
from rwtd_sam3.eval.runner import (
    PROMPT_EVAL_DEFAULT_BATCH_SIZE,
    PROMPT_EVAL_DEFAULT_PREFETCH_FACTOR,
    inspect_dataset,
    parse_dataset_partition_spec,
    run_evaluation,
    run_predict_one,
    set_global_seed,
)
from rwtd_sam3.eval.architexture_binary import (
    run_architexture_binary_evaluation,
    run_architexture_binary_predict_one,
)
from rwtd_sam3.eval.detexture_binary import (
    run_detexture_binary_evaluation,
    run_detexture_binary_predict_one,
)
from rwtd_sam3.eval.detexture_multi import (
    run_detexture_multi_evaluation,
    run_detexture_multi_predict_one,
)
from rwtd_sam3.eval.cstd_binary import (
    run_cstd_binary_evaluation,
    run_cstd_binary_predict_one,
)
from rwtd_sam3.eval.glas_autosam import (
    GLAS_AUTOSAM_REPRODUCTION_PROFILE,
    run_glas_autosam_eval,
    run_glas_autosam_train,
)
from rwtd_sam3.eval.glas_binary import (
    run_glas_binary_evaluation,
    run_glas_binary_predict_one,
)
from rwtd_sam3.eval.glas_frozen_feature_mask_head import (
    run_glas_frozen_mask_head_eval,
    run_glas_frozen_mask_head_train,
)
from rwtd_sam3.eval.monuseg_frozen_feature_mask_head import (
    GLAS_FROZEN_MASK_HEAD_SELECTION_CHECKPOINT_MODES,
    GLAS_FROZEN_MASK_HEAD_SELECTION_METRICS,
    run_monuseg_frozen_mask_head_eval,
    run_monuseg_frozen_mask_head_train,
)
from rwtd_sam3.eval.stld_frozen_feature_mask_head import (
    run_stld_frozen_mask_head_eval,
    run_stld_frozen_mask_head_train,
)
from rwtd_sam3.eval.caid_frozen_feature_mask_head import (
    run_caid_frozen_mask_head_eval,
    run_caid_frozen_mask_head_train,
)
from rwtd_sam3.eval.rwtd_frozen_feature_mask_head import (
    run_rwtd_frozen_mask_head_eval,
    run_rwtd_frozen_mask_head_train,
)
from rwtd_sam3.eval.yknd_frozen_feature_mask_head import (
    run_yknd_frozen_mask_head_eval,
    run_yknd_frozen_mask_head_train,
)
from rwtd_sam3.eval.stld_autosam import (
    STLD_AUTOSAM_AUGMENTATION_POLICIES,
    run_stld_autosam_eval,
    run_stld_autosam_train,
)
from rwtd_sam3.eval.rwtd_autosam import run_rwtd_autosam_eval, run_rwtd_autosam_train
from rwtd_sam3.eval.monuseg_autosam import (
    MONUSEG_AUTOSAM_AUGMENTATION_POLICIES,
    MONUSEG_AUTOSAM_REPRODUCTION_PROFILE,
    MONUSEG_AUTOSAM_SELECTION_METRICS,
    run_monuseg_autosam_eval,
    run_monuseg_autosam_train,
)
from rwtd_sam3.eval.experiment_registry import (
    CROSS_DATASET_EXPERIMENT_VARIANTS,
    DEFAULT_CROSS_DATASET_EXPERIMENT,
)
from rwtd_sam3.eval.sam2_baseline import run_sam2_evaluation, run_sam2_predict_one
from rwtd_sam3.eval.sam2_official_rwtd import run_sam2_official_evaluation, run_sam2_official_predict_one
from rwtd_sam3.eval.sam3_auto import run_sam3_auto_evaluation, run_sam3_auto_predict_one
from rwtd_sam3.eval.coarse_vs_fine_sam_scales import (
    run_coarse_vs_fine_sam_probe_eval,
    run_coarse_vs_fine_sam_probe_train,
)
from rwtd_sam3.eval.coarse_vs_fine_linear_probe import (
    run_coarse_vs_fine_linear_probe_eval,
    run_coarse_vs_fine_linear_probe_train,
)
from rwtd_sam3.models.sam2_runner import DEFAULT_SAM2_MODEL_ID
from rwtd_sam3.models.sam3_auto_runner import DEFAULT_SAM3_AUTO_MODEL_ID
from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID
from rwtd_sam3.models.sam3_feature_cluster_multiregion_runner import (
    DETEXTURE_MULTI_DEFAULT_VARIANT,
    DETEXTURE_MULTI_SUPPORTED_VARIANTS,
)
from rwtd_sam3.models.sam3_coarse_vs_fine_scale_probe import STAGE2_VARIANT_SPECS

FROZEN_SAM_HEAD_ALLOWED_MODEL_ID = "facebook/sam3"
from rwtd_sam3.models.sam3_coarse_vs_fine_linear_probe import LINEAR_PROBE_VARIANT_SPECS
from rwtd_sam3.models.sam3_frozen_multiscale_mask_head import (
    FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS,
    FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS,
    FROZEN_MASK_HEAD_LOSS_VARIANTS,
    FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS,
    FROZEN_MASK_HEAD_VARIANT_SPECS,
    RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD,
    RESIDUAL_HEAD_GATE_MODES,
    is_coarse_plus_residual_variant,
    is_multibank_null_refine_variant,
)
from rwtd_sam3.data.monuseg_binary import MONUSEG_HF_DATASET_NAME
from rwtd_sam3.eval.frozen_mask_head_augmentations import FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES
from rwtd_sam3.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)

DEFAULT_STLD_AUTOSAM_CHECKPOINT_PATH = str(Path("checkpoints") / "sam_vit_h.pth")


def build_parser() -> argparse.ArgumentParser:
    """Create the root CLI parser for dataset inspection and evaluation."""

    parser = argparse.ArgumentParser(description="RWTD dataset inspection and SAM evaluation.")
    parser.add_argument("--log-level", default="INFO", help="Python log level.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible local behavior.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect RWTD dataset metadata.")
    add_dataset_args(inspect_parser)
    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of preview points per oracle field to include in the printed sample summary.",
    )

    predict_parser = subparsers.add_parser("predict-one", help="Run one RWTD sample through SAM 3.")
    add_dataset_args(predict_parser)
    add_model_args(predict_parser)
    predict_parser.add_argument("--index", type=int, required=True, help="Row index within the selected split.")
    predict_parser.add_argument(
        "--protocol",
        choices=("text", "oracle_points", "both"),
        required=True,
        help="Prompting protocol to run.",
    )
    predict_parser.add_argument("--output-dir", default=None, help="Output directory for prediction artifacts.")
    add_save_visuals_args(predict_parser, "Save prediction visualization panels.")

    eval_parser = subparsers.add_parser("eval", help="Evaluate SAM 3 on RWTD.")
    add_dataset_args(eval_parser)
    add_model_args(eval_parser)
    eval_parser.add_argument(
        "--protocol",
        choices=("text", "oracle_points", "both"),
        required=True,
        help="Prompting protocol to run.",
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    eval_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    eval_parser.add_argument(
        "--wandb-project",
        default="rwtd-sam3",
        help="Weights & Biases project name.",
    )
    eval_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit Weights & Biases run name.",
    )
    eval_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_eval_loader_args(eval_parser)
    add_save_visuals_args(eval_parser, "Save visualization panels during eval.")

    predict_sam2_parser = subparsers.add_parser("predict-sam2", help="Run one RWTD sample through the SAM-2 baseline.")
    add_dataset_args(predict_sam2_parser)
    add_sam2_model_args(predict_sam2_parser)
    predict_sam2_parser.set_defaults(split="test")
    predict_sam2_parser.add_argument("--index", type=int, required=True, help="Row index within the selected split.")
    predict_sam2_parser.add_argument(
        "--variant",
        choices=("sam2", "sam2_star", "both"),
        default="sam2_star",
        help="SAM-2 baseline variant to run.",
    )
    predict_sam2_parser.add_argument("--output-dir", default=None, help="Output directory for prediction artifacts.")
    add_save_visuals_args(predict_sam2_parser, "Save prediction visualization panels.")

    eval_sam2_parser = subparsers.add_parser("eval-sam2", help="Evaluate TextureSAM's SAM-2 baselines on RWTD.")
    add_dataset_args(eval_sam2_parser)
    add_sam2_model_args(eval_sam2_parser)
    eval_sam2_parser.set_defaults(split="test")
    eval_sam2_parser.add_argument(
        "--variant",
        choices=("sam2", "sam2_star", "both"),
        default="sam2_star",
        help="SAM-2 baseline variant to run.",
    )
    eval_sam2_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_sam2_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    eval_sam2_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_sam2_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_sam2_parser.add_argument(
        "--wandb-project",
        default="rwtd-sam2",
        help="Weights & Biases project name.",
    )
    eval_sam2_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_sam2_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_save_visuals_args(eval_sam2_parser, "Save visualization panels during eval.")

    predict_sam2_official_parser = subparsers.add_parser(
        "predict-sam2-official",
        help="Run one official TextureSAM Kaust256 RWTD sample through the SAM-2 baseline.",
    )
    add_sam2_model_args(predict_sam2_official_parser)
    add_kaust256_args(predict_sam2_official_parser)
    predict_sam2_official_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Natural-sorted row index within the local Kaust256 dataset.",
    )
    predict_sam2_official_parser.add_argument(
        "--variant",
        choices=("sam2", "sam2_star", "both"),
        default="sam2_star",
        help="SAM-2 baseline variant to run.",
    )
    predict_sam2_official_parser.add_argument("--output-dir", default=None, help="Output directory for prediction artifacts.")
    add_save_visuals_args(predict_sam2_official_parser, "Save prediction visualization panels.")

    eval_sam2_official_parser = subparsers.add_parser(
        "eval-sam2-official",
        help="Evaluate SAM-2 on the official TextureSAM Kaust256 RWTD dataset.",
    )
    add_sam2_model_args(eval_sam2_official_parser)
    add_kaust256_args(eval_sam2_official_parser)
    eval_sam2_official_parser.add_argument(
        "--variant",
        choices=("sam2", "sam2_star", "both"),
        default="sam2_star",
        help="SAM-2 baseline variant to run.",
    )
    eval_sam2_official_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_sam2_official_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    eval_sam2_official_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_sam2_official_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_sam2_official_parser.add_argument(
        "--wandb-project",
        default="rwtd-sam2-official",
        help="Weights & Biases project name.",
    )
    eval_sam2_official_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_sam2_official_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_save_visuals_args(eval_sam2_official_parser, "Save visualization panels during eval.")

    predict_sam3_auto_parser = subparsers.add_parser(
        "predict-sam3-auto",
        help="Run one RWTD sample through the SAM-3 automatic mask comparison.",
    )
    add_dataset_args(predict_sam3_auto_parser)
    add_sam3_auto_model_args(predict_sam3_auto_parser)
    predict_sam3_auto_parser.set_defaults(split="test")
    predict_sam3_auto_parser.add_argument("--index", type=int, required=True, help="Row index within the selected split.")
    predict_sam3_auto_parser.add_argument(
        "--variant",
        choices=(
            "default",
            "dense",
            "feature_mask",
            "feature_cluster_global",
            "feature_cluster_coarse_to_fine_global",
            "feature_cluster_coarse_to_fine_global_pooled_init",
            "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only",
            "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
            "boundary_refine_sweep",
            "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only",
            "flip_avg_plus_edge_debias",
            "feature_cluster_coarse_to_fine_global_pooled_init_direct",
            "mask_prompt_invariance_control",
            "both",
        ),
        default="dense",
        help="SAM-3 automatic-mask variant to run.",
    )
    predict_sam3_auto_parser.add_argument("--output-dir", default=None, help="Output directory for prediction artifacts.")
    add_save_visuals_args(predict_sam3_auto_parser, "Save prediction visualization panels.")

    eval_sam3_auto_parser = subparsers.add_parser(
        "eval-sam3-auto",
        help="Evaluate SAM-3 automatic mask generation on RWTD with the same scoring protocol.",
    )
    add_dataset_args(eval_sam3_auto_parser)
    add_sam3_auto_model_args(eval_sam3_auto_parser)
    eval_sam3_auto_parser.set_defaults(split="test")
    eval_sam3_auto_parser.add_argument(
        "--variant",
        choices=(
            "default",
            "dense",
            "feature_mask",
            "feature_cluster_global",
            "feature_cluster_coarse_to_fine_global",
            "feature_cluster_coarse_to_fine_global_pooled_init",
            "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only",
            "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
            "boundary_refine_sweep",
            "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only",
            "flip_avg_plus_edge_debias",
            "feature_cluster_coarse_to_fine_global_pooled_init_direct",
            "mask_prompt_invariance_control",
            "both",
        ),
        default="dense",
        help="SAM-3 automatic-mask variant to run.",
    )
    eval_sam3_auto_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_sam3_auto_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    eval_sam3_auto_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_sam3_auto_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_sam3_auto_parser.add_argument(
        "--wandb-project",
        default="rwtd-sam3-auto",
        help="Weights & Biases project name.",
    )
    eval_sam3_auto_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_sam3_auto_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_dataset_partition_args(eval_sam3_auto_parser)
    add_save_visuals_args(eval_sam3_auto_parser, "Save visualization panels during eval.")

    train_coarse_vs_fine_probe_parser = subparsers.add_parser(
        "train-coarse-vs-fine-sam-probe",
        help="Train the tiny Stage-2 coarse-vs-fine SAM scale-gated probe on frozen RWTD, official-split ArchiTexture, or official-split CSTD SAM features.",
    )
    add_coarse_vs_fine_stage2_dataset_args(train_coarse_vs_fine_probe_parser)
    add_sam3_auto_model_args(train_coarse_vs_fine_probe_parser)
    train_coarse_vs_fine_probe_parser.add_argument(
        "--variant",
        choices=tuple(STAGE2_VARIANT_SPECS),
        default="learned_global_gates_all_scales",
        help="Stage-2 coarse-vs-fine probe variant to train.",
    )
    train_coarse_vs_fine_probe_parser.add_argument(
        "--train-split",
        choices=("train", "test"),
        default="train",
        help="Dataset split used for probe training. Non-RWTD Stage-2 runs require official split-aware local roots.",
    )
    train_coarse_vs_fine_probe_parser.add_argument(
        "--eval-split",
        choices=("train", "test"),
        default="test",
        help="Dataset split used for post-training evaluation. Non-RWTD Stage-2 runs require official split-aware local roots.",
    )
    train_coarse_vs_fine_probe_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on cached training samples. Kept for backward compatibility with older Stage-2 runs.",
    )
    train_coarse_vs_fine_probe_parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Explicit few-shot training cap. When set, only this many training samples are loaded from --train-split.",
    )
    train_coarse_vs_fine_probe_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on cached evaluation samples.",
    )
    add_coarse_vs_fine_stage2_probe_args(train_coarse_vs_fine_probe_parser)
    train_coarse_vs_fine_probe_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for checkpoints and evaluation artifacts.",
    )
    add_save_visuals_args(train_coarse_vs_fine_probe_parser, "Save evaluation visualization panels after training.")

    eval_coarse_vs_fine_probe_parser = subparsers.add_parser(
        "eval-coarse-vs-fine-sam-probe",
        help="Evaluate a trained Stage-2 coarse-vs-fine SAM scale-gated probe checkpoint on RWTD, official-split ArchiTexture, or official-split CSTD.",
    )
    add_coarse_vs_fine_stage2_dataset_args(eval_coarse_vs_fine_probe_parser)
    add_sam3_auto_model_args(eval_coarse_vs_fine_probe_parser)
    eval_coarse_vs_fine_probe_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a Stage-2 probe checkpoint produced by train-coarse-vs-fine-sam-probe.",
    )
    eval_coarse_vs_fine_probe_parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split used for checkpoint evaluation. Non-RWTD Stage-2 runs require official split-aware local roots.",
    )
    eval_coarse_vs_fine_probe_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated samples.",
    )
    add_coarse_vs_fine_stage2_probe_args(eval_coarse_vs_fine_probe_parser)
    eval_coarse_vs_fine_probe_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_save_visuals_args(eval_coarse_vs_fine_probe_parser, "Save evaluation visualization panels.")

    train_coarse_vs_fine_linear_probe_parser = subparsers.add_parser(
        "train-coarse-vs-fine-linear-probe",
        help="Train a few-shot linear probe on frozen coarse-vs-fine SAM features for RWTD, official-split ArchiTexture, or official-split CSTD.",
    )
    add_coarse_vs_fine_stage2_dataset_args(train_coarse_vs_fine_linear_probe_parser)
    add_sam3_auto_model_args(train_coarse_vs_fine_linear_probe_parser)
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--variant",
        choices=tuple(LINEAR_PROBE_VARIANT_SPECS),
        default="concat_all_scales",
        help="Few-shot linear-probe variant to train.",
    )
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--train-split",
        choices=("train", "test"),
        default="train",
        help="Dataset split used for linear-probe training.",
    )
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--eval-split",
        choices=("train", "test"),
        default="test",
        help="Dataset split used for post-training evaluation.",
    )
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on cached training samples. Kept for backward compatibility with older few-shot wrappers.",
    )
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Explicit few-shot training cap. When set, only this many samples are loaded from --train-split.",
    )
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on post-training evaluation samples.",
    )
    add_coarse_vs_fine_linear_probe_args(train_coarse_vs_fine_linear_probe_parser)
    train_coarse_vs_fine_linear_probe_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for checkpoints and evaluation artifacts.",
    )
    add_save_visuals_args(
        train_coarse_vs_fine_linear_probe_parser,
        "Save evaluation visualization panels after training.",
    )

    eval_coarse_vs_fine_linear_probe_parser = subparsers.add_parser(
        "eval-coarse-vs-fine-linear-probe",
        help="Evaluate a saved few-shot linear probe checkpoint on RWTD, official-split ArchiTexture, or official-split CSTD.",
    )
    add_coarse_vs_fine_stage2_dataset_args(eval_coarse_vs_fine_linear_probe_parser)
    add_sam3_auto_model_args(eval_coarse_vs_fine_linear_probe_parser)
    eval_coarse_vs_fine_linear_probe_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a linear-probe checkpoint produced by train-coarse-vs-fine-linear-probe.",
    )
    eval_coarse_vs_fine_linear_probe_parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split used for checkpoint evaluation.",
    )
    eval_coarse_vs_fine_linear_probe_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated samples.",
    )
    add_coarse_vs_fine_linear_probe_args(eval_coarse_vs_fine_linear_probe_parser)
    eval_coarse_vs_fine_linear_probe_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_save_visuals_args(eval_coarse_vs_fine_linear_probe_parser, "Save evaluation visualization panels.")

    train_glas_frozen_mask_head_parser = subparsers.add_parser(
        "train-glas-frozen-sam-mask-head",
        help="Train a tiny dense-supervised mask head on frozen SAM multiscale features for GlaS.",
    )
    add_glas_binary_root_arg(train_glas_frozen_mask_head_parser)
    add_sam3_auto_model_args(train_glas_frozen_mask_head_parser)
    train_glas_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Frozen SAM scale subset used by the tiny GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--projection-dim",
        type=int,
        default=64,
        help="Per-level 1x1 projection width for the frozen multiscale GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--decoder-dim",
        type=int,
        default=64,
        help="Decoder width for the frozen multiscale GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--memory-token-count",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]),
        help="Number of learned global memory tokens used by the GlaS memory-attention variants.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--attention-heads",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]),
        help="Number of heads used by the GlaS memory-attention variants.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--attention-blocks",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]),
        help="Number of stacked memory-attention or matched-control blocks in the GlaS attention head family.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--train-split",
        choices=("train", "test", "all"),
        default="train",
        help="GlaS split used for optimization.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--eval-split",
        choices=("train", "test", "all"),
        default="test",
        help="GlaS split used for post-training evaluation.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training samples loaded from the selected train split.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--train-subset-manifest",
        default=None,
        help="Optional JSON manifest selecting an explicit GlaS train subset by image ID. Mutually exclusive with --train-limit.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation samples loaded from the selected eval split.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate for the tiny GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for the tiny GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs for the tiny GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight applied to BCEWithLogits in the GlaS mask-head loss.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight applied to Dice loss in the GlaS mask-head loss.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert foreground logits into a binary gland mask at eval time.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--backbone-preprocessing-mode",
        choices=("native_resolution", "autosam_style_1024"),
        default="native_resolution",
        help="Geometry fed into the frozen SAM backbone. native_resolution keeps decoded GlaS sizes; autosam_style_1024 mirrors upstream AutoSAM ResizeLongestSide(1024) before frozen feature extraction.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--resize-height",
        type=int,
        default=512,
        help="Optional explicit image and mask resize height applied before frozen feature extraction, supervision, and eval.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--resize-width",
        type=int,
        default=512,
        help="Optional explicit image and mask resize width applied before frozen feature extraction, supervision, and eval.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training and evaluation artifacts.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--wandb-project",
        default="glas-frozen-sam3",
        help="Weights & Biases project name.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit Weights & Biases run name.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--train-augmentation-policy",
        choices=FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES,
        default="none",
        help="Optional train-only augmentation policy applied before frozen SAM feature extraction.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=None,
        help="Optional eval cadence in epochs for post-hoc Dice convergence tracking on --eval-split.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--selection-split",
        choices=("train", "val"),
        default="train",
        help="Split used for checkpoint selection during per-epoch eval tracking. `val` is only valid when an explicit validation holdout is requested.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--selection-metric",
        choices=GLAS_FROZEN_MASK_HEAD_SELECTION_METRICS,
        default="direct_foreground_dice",
        help="Metric used for checkpoint selection during per-epoch eval tracking.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--selection-checkpoint-mode",
        choices=GLAS_FROZEN_MASK_HEAD_SELECTION_CHECKPOINT_MODES,
        default="final",
        help="Which checkpoint view should be treated as the headline run artifact.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--loss-variant",
        choices=FROZEN_MASK_HEAD_LOSS_VARIANTS,
        default="bce_dice",
        help="Binary segmentation loss variant for the frozen GlaS mask head.",
    )
    train_glas_frozen_mask_head_parser.add_argument(
        "--boundary-weight",
        type=float,
        default=float(FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS["boundary_weight"]),
        help="Extra BCE pixel weight applied on boundary pixels when --loss-variant boundary_weighted_bce_dice is used.",
    )
    add_residual_head_experiment_args(
        train_glas_frozen_mask_head_parser,
        include_training_losses=True,
        include_eval_alpha_override=False,
    )
    add_save_visuals_args(
        train_glas_frozen_mask_head_parser,
        "Save visualization panels during post-training evaluation.",
    )

    eval_glas_frozen_mask_head_parser = subparsers.add_parser(
        "eval-glas-frozen-sam-mask-head",
        help="Evaluate a trained frozen-SAM GlaS mask-head checkpoint on one GlaS split.",
    )
    add_glas_binary_args(eval_glas_frozen_mask_head_parser)
    add_sam3_auto_model_args(eval_glas_frozen_mask_head_parser)
    eval_glas_frozen_mask_head_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint produced by train-glas-frozen-sam-mask-head.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Checkpoint variant placeholder; the stored checkpoint variant overrides this at runtime.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated rows.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert foreground logits into a binary gland mask at eval time. Defaults to the checkpoint value when present.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--memory-token-count",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]),
        help="Checkpoint placeholder; the stored checkpoint memory-token count overrides this at runtime.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--attention-heads",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]),
        help="Checkpoint placeholder; the stored checkpoint attention-head count overrides this at runtime.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--attention-blocks",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]),
        help="Checkpoint placeholder; the stored checkpoint attention-block count overrides this at runtime.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--backbone-preprocessing-mode",
        choices=("native_resolution", "autosam_style_1024"),
        default="native_resolution",
        help="Geometry fed into the frozen SAM backbone. When the checkpoint stores a non-native mode, that stored mode overrides this placeholder.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--resize-height",
        type=int,
        default=512,
        help="Optional explicit eval-time resize height. If the checkpoint already stores a resize protocol, this must match it.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--resize-width",
        type=int,
        default=512,
        help="Optional explicit eval-time resize width. If the checkpoint already stores a resize protocol, this must match it.",
    )
    eval_glas_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_residual_head_experiment_args(
        eval_glas_frozen_mask_head_parser,
        include_training_losses=False,
        include_eval_alpha_override=True,
    )
    add_dataset_partition_args(eval_glas_frozen_mask_head_parser)
    add_save_visuals_args(eval_glas_frozen_mask_head_parser, "Save visualization panels during eval.")

    train_monuseg_frozen_mask_head_parser = subparsers.add_parser(
        "train-monuseg-frozen-sam-mask-head",
        help="Train a tiny dense-supervised mask head on frozen SAM multiscale features for MoNuSeg.",
    )
    add_monuseg_binary_args(train_monuseg_frozen_mask_head_parser)
    add_sam3_auto_model_args(train_monuseg_frozen_mask_head_parser)
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Frozen SAM scale subset used by the tiny MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--projection-dim",
        type=int,
        default=64,
        help="Per-level 1x1 projection width for the frozen multiscale MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--decoder-dim",
        type=int,
        default=64,
        help="Decoder width for the frozen multiscale MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--memory-token-count",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]),
        help="Number of learned global memory tokens used by the MoNuSeg memory-attention variants.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--attention-heads",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]),
        help="Number of heads used by the MoNuSeg memory-attention variants.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--attention-blocks",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]),
        help="Number of stacked memory-attention or matched-control blocks in the MoNuSeg attention head family.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--train-split",
        choices=("train", "test", "all"),
        default="train",
        help="MoNuSeg split used for optimization. The default SAM3 historical contract uses the full official train split unless an explicit holdout is requested.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--eval-split",
        choices=("train", "test", "all"),
        default="test",
        help="MoNuSeg split used for post-training evaluation.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--validation-holdout-count",
        type=int,
        default=0,
        help="Explicit number of official MoNuSeg train images held out for checkpoint selection. Zero disables the internal holdout split.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--validation-subset-seed",
        type=int,
        default=0,
        help="Seed used to choose the explicit MoNuSeg validation holdout subset.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training samples loaded from the selected train split.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--train-subset-manifest",
        default=None,
        help="Optional JSON manifest selecting an explicit MoNuSeg train subset by patient ID. Mutually exclusive with --train-limit.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation samples loaded from the selected eval split.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate for the tiny MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for the tiny MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs for the tiny MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight applied to BCEWithLogits in the MoNuSeg mask-head loss.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight applied to Dice loss in the MoNuSeg mask-head loss.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert foreground logits into a binary nucleus mask at eval time.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--backbone-preprocessing-mode",
        choices=("native_resolution", "autosam_style_1024"),
        default="native_resolution",
        help="Geometry fed into the frozen SAM backbone. native_resolution keeps decoded MoNuSeg sizes; autosam_style_1024 mirrors upstream AutoSAM ResizeLongestSide(1024) before frozen feature extraction.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--resize-height",
        type=int,
        default=512,
        help="Optional explicit image and mask resize height applied before frozen feature extraction, supervision, and eval.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--resize-width",
        type=int,
        default=512,
        help="Optional explicit image and mask resize width applied before frozen feature extraction, supervision, and eval.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training and evaluation artifacts.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--wandb-project",
        default="monuseg-frozen-sam3",
        help="Weights & Biases project name.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit Weights & Biases run name.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--train-augmentation-policy",
        choices=FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES,
        default="none",
        help="Optional train-only augmentation policy applied before frozen SAM feature extraction.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=None,
        help="Optional eval cadence in epochs for post-hoc Dice convergence tracking on --eval-split.",
    )
    add_eval_loader_args(train_monuseg_frozen_mask_head_parser)
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--selection-split",
        choices=("train", "val"),
        default="train",
        help="Split used for checkpoint selection during per-epoch eval tracking. `val` is only valid when an explicit validation holdout is requested.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--selection-metric",
        choices=GLAS_FROZEN_MASK_HEAD_SELECTION_METRICS,
        default="direct_foreground_dice",
        help="Metric used for checkpoint selection during per-epoch eval tracking.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--selection-checkpoint-mode",
        choices=GLAS_FROZEN_MASK_HEAD_SELECTION_CHECKPOINT_MODES,
        default="final",
        help="Which checkpoint view should be treated as the headline run artifact.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--upstream-eval-image-size",
        type=int,
        default=int(MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["prompt_image_size"]),
        help="AutoSAM-style resized-binary eval size used for upstream_eval_iou/dice on the frozen-head route.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--loss-variant",
        choices=FROZEN_MASK_HEAD_LOSS_VARIANTS,
        default="bce_dice",
        help="Binary segmentation loss variant for the frozen MoNuSeg mask head.",
    )
    train_monuseg_frozen_mask_head_parser.add_argument(
        "--boundary-weight",
        type=float,
        default=float(FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS["boundary_weight"]),
        help="Extra BCE pixel weight applied on boundary pixels when --loss-variant boundary_weighted_bce_dice is used.",
    )
    add_residual_head_experiment_args(
        train_monuseg_frozen_mask_head_parser,
        include_training_losses=True,
        include_eval_alpha_override=False,
    )
    add_save_visuals_args(
        train_monuseg_frozen_mask_head_parser,
        "Save visualization panels during post-training evaluation.",
    )

    eval_monuseg_frozen_mask_head_parser = subparsers.add_parser(
        "eval-monuseg-frozen-sam-mask-head",
        help="Evaluate a trained frozen-SAM MoNuSeg mask-head checkpoint on one MoNuSeg split.",
    )
    add_monuseg_binary_args(eval_monuseg_frozen_mask_head_parser)
    add_sam3_auto_model_args(eval_monuseg_frozen_mask_head_parser)
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint produced by train-monuseg-frozen-sam-mask-head.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Checkpoint variant placeholder; the stored checkpoint variant overrides this at runtime.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated rows.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert foreground logits into a binary nucleus mask at eval time. Defaults to the checkpoint value when present.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--memory-token-count",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]),
        help="Checkpoint placeholder; the stored checkpoint memory-token count overrides this at runtime.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--attention-heads",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]),
        help="Checkpoint placeholder; the stored checkpoint attention-head count overrides this at runtime.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--attention-blocks",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]),
        help="Checkpoint placeholder; the stored checkpoint attention-block count overrides this at runtime.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--backbone-preprocessing-mode",
        choices=("native_resolution", "autosam_style_1024"),
        default="native_resolution",
        help="Geometry fed into the frozen SAM backbone. Defaults to the checkpoint value when present.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--resize-height",
        type=int,
        default=512,
        help="Optional explicit eval-time resize height. If the checkpoint already stores a resize protocol, this must match it.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--resize-width",
        type=int,
        default=512,
        help="Optional explicit eval-time resize width. If the checkpoint already stores a resize protocol, this must match it.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--upstream-eval-image-size",
        type=int,
        default=int(MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["prompt_image_size"]),
        help="AutoSAM-style resized-binary eval size used for upstream_eval_iou/dice. Defaults to the checkpoint value when present.",
    )
    eval_monuseg_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_residual_head_experiment_args(
        eval_monuseg_frozen_mask_head_parser,
        include_training_losses=False,
        include_eval_alpha_override=True,
    )
    add_dataset_partition_args(eval_monuseg_frozen_mask_head_parser)
    add_save_visuals_args(eval_monuseg_frozen_mask_head_parser, "Save visualization panels during eval.")

    train_yknd_frozen_mask_head_parser = subparsers.add_parser(
        "train-yknd-frozen-sam-mask-head",
        help="Train a tiny dense-supervised mask head on frozen SAM multiscale features for YKND.",
    )
    add_yknd_args(train_yknd_frozen_mask_head_parser)
    add_sam3_auto_model_args(train_yknd_frozen_mask_head_parser)
    train_yknd_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Frozen SAM scale subset used by the tiny YKND mask head.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--projection-dim",
        type=int,
        default=64,
        help="Per-level 1x1 projection width for the frozen multiscale YKND mask head.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--decoder-dim",
        type=int,
        default=64,
        help="Decoder width for the frozen multiscale YKND mask head.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--train-split",
        choices=("train", "test", "all"),
        default="train",
        help="YKND split used for optimization.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--eval-split",
        choices=("train", "test", "all"),
        default="test",
        help="YKND split used for post-training evaluation.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training samples.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation samples.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate for the tiny YKND mask head.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for the tiny YKND mask head.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs for the tiny YKND mask head.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight applied to BCEWithLogits in the YKND mask-head loss.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight applied to Dice loss in the YKND mask-head loss.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.0,
        help="Extra BCE pixel weight applied on boundary pixels.",
    )
    train_yknd_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training and evaluation artifacts.",
    )
    add_residual_head_experiment_args(
        train_yknd_frozen_mask_head_parser,
        include_training_losses=True,
        include_eval_alpha_override=False,
    )

    eval_yknd_frozen_mask_head_parser = subparsers.add_parser(
        "eval-yknd-frozen-sam-mask-head",
        help="Evaluate a trained frozen-SAM YKND mask-head checkpoint.",
    )
    add_yknd_args(eval_yknd_frozen_mask_head_parser)
    add_sam3_auto_model_args(eval_yknd_frozen_mask_head_parser)
    eval_yknd_frozen_mask_head_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint produced by train-yknd-frozen-sam-mask-head.",
    )
    eval_yknd_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Checkpoint variant placeholder.",
    )
    eval_yknd_frozen_mask_head_parser.add_argument("--projection-dim", type=int, default=64)
    eval_yknd_frozen_mask_head_parser.add_argument("--decoder-dim", type=int, default=64)
    eval_yknd_frozen_mask_head_parser.add_argument("--attention-hidden-dim", type=int, default=32)
    eval_yknd_frozen_mask_head_parser.add_argument("--cross-attn-query-stride", type=int, default=2)
    eval_yknd_frozen_mask_head_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated rows.",
    )
    eval_yknd_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )

    train_stld_frozen_mask_head_parser = subparsers.add_parser(
        "train-stld-frozen-sam-mask-head",
        help="Train a tiny dense-supervised mask head on frozen SAM multiscale features for STLD.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the split-aware STLD root with ImageSets/Segmentation/{train,test}.txt plus images/ and labels/.",
    )
    add_sam3_auto_model_args(train_stld_frozen_mask_head_parser)
    train_stld_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Frozen SAM scale subset used by the tiny STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--projection-dim",
        type=int,
        default=64,
        help="Per-level 1x1 projection width for the frozen multiscale STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--decoder-dim",
        type=int,
        default=64,
        help="Decoder width for the frozen multiscale STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--memory-token-count",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]),
        help="Number of learned global memory tokens used by the tiny STLD memory-attention head variants.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--attention-heads",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]),
        help="Number of heads used by the STLD memory-attention variants.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--attention-blocks",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]),
        help="Number of stacked memory-attention or matched-control blocks in the STLD attention head family.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--train-split",
        choices=("train", "test"),
        default="train",
        help="STLD split used for optimization.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--eval-split",
        choices=("train", "test"),
        default="test",
        help="STLD split used for post-training evaluation.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training samples loaded from the selected train split.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--train-subset-manifest",
        default=None,
        help="Optional JSON manifest selecting an explicit STLD train subset by image ID. Mutually exclusive with --train-limit.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation samples loaded from the selected eval split.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate for the tiny STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for the tiny STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs for the tiny STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight applied to BCEWithLogits in the STLD mask-head loss.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight applied to Dice loss in the STLD mask-head loss.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert foreground logits into a binary foreground mask at eval time.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional explicit image and mask resize height applied before frozen feature extraction, supervision, and eval.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional explicit image and mask resize width applied before frozen feature extraction, supervision, and eval.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--backbone-preprocessing-mode",
        choices=("native_resolution", "autosam_style_1024"),
        default="native_resolution",
        help="Geometry fed into the frozen SAM backbone before STLD feature extraction.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training and evaluation artifacts.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--train-augmentation-policy",
        choices=FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES,
        default="none",
        help="Optional train-only augmentation policy applied before frozen SAM feature extraction.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=None,
        help="Optional eval cadence in epochs for post-hoc Dice convergence tracking on --eval-split.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--loss-variant",
        choices=FROZEN_MASK_HEAD_LOSS_VARIANTS,
        default="bce_dice",
        help="Binary segmentation loss variant for the frozen STLD mask head.",
    )
    train_stld_frozen_mask_head_parser.add_argument(
        "--boundary-weight",
        type=float,
        default=float(FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS["boundary_weight"]),
        help="Extra BCE pixel weight applied on boundary pixels when --loss-variant boundary_weighted_bce_dice is used.",
    )
    add_residual_head_experiment_args(
        train_stld_frozen_mask_head_parser,
        include_training_losses=True,
        include_eval_alpha_override=False,
    )
    add_save_visuals_args(
        train_stld_frozen_mask_head_parser,
        "Save visualization panels during post-training evaluation.",
    )

    eval_stld_frozen_mask_head_parser = subparsers.add_parser(
        "eval-stld-frozen-sam-mask-head",
        help="Evaluate a trained frozen-SAM STLD mask-head checkpoint on one STLD split.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the split-aware STLD root with ImageSets/Segmentation/{train,test}.txt plus images/ and labels/.",
    )
    add_sam3_auto_model_args(eval_stld_frozen_mask_head_parser)
    eval_stld_frozen_mask_head_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint produced by train-stld-frozen-sam-mask-head.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="STLD split used for checkpoint evaluation.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--variant",
        choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS),
        default="all_scales",
        help="Checkpoint variant placeholder; the stored checkpoint variant overrides this at runtime.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated rows.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert foreground logits into a binary foreground mask at eval time. Defaults to the checkpoint value when present.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--memory-token-count",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]),
        help="Checkpoint placeholder; the stored checkpoint memory-token count overrides this at runtime.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--attention-heads",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]),
        help="Checkpoint placeholder; the stored checkpoint attention-head count overrides this at runtime.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--attention-blocks",
        type=int,
        default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]),
        help="Checkpoint placeholder; the stored checkpoint attention-block count overrides this at runtime.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional explicit eval-time resize height. If the checkpoint already stores a resize protocol, this must match it.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional explicit eval-time resize width. If the checkpoint already stores a resize protocol, this must match it.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--backbone-preprocessing-mode",
        choices=("native_resolution", "autosam_style_1024"),
        default="native_resolution",
        help="Geometry fed into the frozen SAM backbone. Defaults to the checkpoint value when present.",
    )
    eval_stld_frozen_mask_head_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_residual_head_experiment_args(
        eval_stld_frozen_mask_head_parser,
        include_training_losses=False,
        include_eval_alpha_override=True,
    )
    add_dataset_partition_args(eval_stld_frozen_mask_head_parser)
    add_save_visuals_args(eval_stld_frozen_mask_head_parser, "Save visualization panels during eval.")

    train_caid_frozen_mask_head_parser = subparsers.add_parser(
        "train-caid-frozen-sam-mask-head",
        parents=[train_stld_frozen_mask_head_parser],
        add_help=False,
        help="Train a tiny dense-supervised mask head on frozen SAM multiscale features for CAID.",
        conflict_handler="resolve",
    )
    eval_caid_frozen_mask_head_parser = subparsers.add_parser(
        "eval-caid-frozen-sam-mask-head",
        parents=[eval_stld_frozen_mask_head_parser],
        add_help=False,
        help="Evaluate a trained frozen-SAM CAID mask-head checkpoint on one CAID split.",
        conflict_handler="resolve",
    )

    train_rwtd_frozen_mask_head_parser = subparsers.add_parser(
        "train-rwtd-frozen-sam-mask-head",
        help="Train a tiny frozen-SAM mask head on RWTD with partition-invariant eval.",
    )
    add_dataset_args(train_rwtd_frozen_mask_head_parser)
    add_sam3_auto_model_args(train_rwtd_frozen_mask_head_parser)
    train_rwtd_frozen_mask_head_parser.add_argument("--variant", choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS), default="all_scales", help="Frozen SAM scale subset used by the tiny RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--projection-dim", type=int, default=64, help="Per-level 1x1 projection width for the frozen multiscale RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--decoder-dim", type=int, default=64, help="Decoder width for the frozen multiscale RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--memory-token-count", type=int, default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]), help="Number of learned global memory tokens used by the RWTD memory-attention variants.")
    train_rwtd_frozen_mask_head_parser.add_argument("--attention-heads", type=int, default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]), help="Number of heads used by the RWTD memory-attention variants.")
    train_rwtd_frozen_mask_head_parser.add_argument("--attention-blocks", type=int, default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]), help="Number of stacked memory-attention or matched-control blocks in the RWTD attention head family.")
    train_rwtd_frozen_mask_head_parser.add_argument("--train-split", choices=("train", "test"), default="train", help="RWTD split used for optimization.")
    train_rwtd_frozen_mask_head_parser.add_argument("--eval-split", choices=("train", "test"), default="test", help="RWTD split used for post-training evaluation.")
    train_rwtd_frozen_mask_head_parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on the number of training samples loaded from the selected train split.")
    train_rwtd_frozen_mask_head_parser.add_argument("--train-subset-manifest", default=None, help="Optional JSON manifest selecting an explicit RWTD train subset by crop name. Mutually exclusive with --train-limit.")
    train_rwtd_frozen_mask_head_parser.add_argument("--eval-limit", type=int, default=None, help="Optional cap on the number of evaluation samples loaded from the selected eval split.")
    train_rwtd_frozen_mask_head_parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate for the tiny RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay for the tiny RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs for the tiny RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--bce-weight", type=float, default=1.0, help="Weight applied to BCEWithLogits in the RWTD mask-head loss.")
    train_rwtd_frozen_mask_head_parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight applied to Dice loss in the RWTD mask-head loss.")
    train_rwtd_frozen_mask_head_parser.add_argument("--foreground-threshold", type=float, default=0.5, help="Probability threshold used to convert foreground logits into a binary mask at eval time.")
    train_rwtd_frozen_mask_head_parser.add_argument("--resize-height", type=int, default=None, help="Optional explicit image and mask resize height applied before frozen feature extraction, supervision, and eval.")
    train_rwtd_frozen_mask_head_parser.add_argument("--resize-width", type=int, default=None, help="Optional explicit image and mask resize width applied before frozen feature extraction, supervision, and eval.")
    train_rwtd_frozen_mask_head_parser.add_argument("--backbone-preprocessing-mode", choices=("native_resolution", "autosam_style_1024"), default="native_resolution", help="Geometry fed into the frozen SAM backbone before RWTD feature extraction.")
    train_rwtd_frozen_mask_head_parser.add_argument("--output-dir", default=None, help="Output directory for training and evaluation artifacts.")
    train_rwtd_frozen_mask_head_parser.add_argument("--train-augmentation-policy", choices=FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES, default="none", help="Optional train-only augmentation policy applied before frozen SAM feature extraction.")
    train_rwtd_frozen_mask_head_parser.add_argument("--eval-every-epochs", type=int, default=None, help="Optional eval cadence in epochs for post-hoc convergence tracking on --eval-split.")
    train_rwtd_frozen_mask_head_parser.add_argument("--loss-variant", choices=FROZEN_MASK_HEAD_LOSS_VARIANTS, default="bce_dice", help="Binary segmentation loss variant for the frozen RWTD mask head.")
    train_rwtd_frozen_mask_head_parser.add_argument("--boundary-weight", type=float, default=float(FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS["boundary_weight"]), help="Extra BCE pixel weight applied on boundary pixels when --loss-variant boundary_weighted_bce_dice is used.")
    add_residual_head_experiment_args(
        train_rwtd_frozen_mask_head_parser,
        include_training_losses=True,
        include_eval_alpha_override=False,
    )
    add_save_visuals_args(train_rwtd_frozen_mask_head_parser, "Save visualization panels during post-training evaluation.")

    eval_rwtd_frozen_mask_head_parser = subparsers.add_parser(
        "eval-rwtd-frozen-sam-mask-head",
        help="Evaluate a trained frozen-SAM RWTD mask-head checkpoint on one RWTD split.",
    )
    add_dataset_args(eval_rwtd_frozen_mask_head_parser)
    add_sam3_auto_model_args(eval_rwtd_frozen_mask_head_parser)
    eval_rwtd_frozen_mask_head_parser.add_argument("--checkpoint-path", required=True, help="Path to a checkpoint produced by train-rwtd-frozen-sam-mask-head.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--variant", choices=tuple(FROZEN_MASK_HEAD_VARIANT_SPECS), default="all_scales", help="Checkpoint variant placeholder; the stored checkpoint variant overrides this at runtime.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--foreground-threshold", type=float, default=0.5, help="Probability threshold used to convert foreground logits into a binary mask at eval time. Defaults to the checkpoint value when present.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--memory-token-count", type=int, default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]), help="Checkpoint placeholder; the stored checkpoint memory-token count overrides this at runtime.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--attention-heads", type=int, default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]), help="Checkpoint placeholder; the stored checkpoint attention-head count overrides this at runtime.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--attention-blocks", type=int, default=int(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]), help="Checkpoint placeholder; the stored checkpoint attention-block count overrides this at runtime.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--resize-height", type=int, default=None, help="Optional explicit eval-time resize height. If the checkpoint already stores a resize protocol, this must match it.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--resize-width", type=int, default=None, help="Optional explicit eval-time resize width. If the checkpoint already stores a resize protocol, this must match it.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--backbone-preprocessing-mode", choices=("native_resolution", "autosam_style_1024"), default="native_resolution", help="Geometry fed into the frozen SAM backbone. Defaults to the checkpoint value when present.")
    eval_rwtd_frozen_mask_head_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    add_residual_head_experiment_args(
        eval_rwtd_frozen_mask_head_parser,
        include_training_losses=False,
        include_eval_alpha_override=True,
    )
    add_dataset_partition_args(eval_rwtd_frozen_mask_head_parser)
    add_save_visuals_args(eval_rwtd_frozen_mask_head_parser, "Save visualization panels during eval.")

    train_stld_autosam_parser = subparsers.add_parser(
        "train-stld-autosam",
        help="Train the upstream AutoSAM prompt-generator route on the prepared STLD split.",
    )
    train_stld_autosam_parser.set_defaults(route="stld")
    train_stld_autosam_parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the split-aware STLD root with ImageSets/Segmentation/{train,test}.txt plus images/ and labels/.",
    )
    train_stld_autosam_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for STLD AutoSAM training and evaluation.",
    )
    train_stld_autosam_parser.add_argument(
        "--sam-checkpoint-path",
        default=DEFAULT_STLD_AUTOSAM_CHECKPOINT_PATH,
        help="Path to the official Meta SAM checkpoint used by AutoSAM.",
    )
    train_stld_autosam_parser.add_argument(
        "--sam-model-type",
        choices=("vit_h", "vit_l", "vit_b"),
        default="vit_h",
        help="SAM model type passed into the upstream AutoSAM SAM builder.",
    )
    train_stld_autosam_parser.add_argument(
        "--prompt-image-size",
        type=int,
        default=256,
        help="Square prompt-generator input size (Idim in the upstream AutoSAM code).",
    )
    train_stld_autosam_parser.add_argument(
        "--autosam-backbone-order",
        type=int,
        default=85,
        help="HarDNet order for the upstream AutoSAM prompt generator.",
    )
    train_stld_autosam_parser.add_argument(
        "--autosam-depth-wise",
        action="store_true",
        help="Enable the upstream AutoSAM depth-wise HarDNet variant.",
    )
    train_stld_autosam_parser.add_argument(
        "--train-split",
        choices=("train", "test"),
        default="train",
        help="STLD split used for optimization.",
    )
    train_stld_autosam_parser.add_argument(
        "--eval-split",
        choices=("train", "test"),
        default="test",
        help="STLD split used for post-training evaluation.",
    )
    train_stld_autosam_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training samples loaded from the selected train split.",
    )
    train_stld_autosam_parser.add_argument(
        "--train-subset-manifest",
        default=None,
        help="Optional JSON manifest selecting an explicit STLD train subset by image ID. Mutually exclusive with --train-limit.",
    )
    train_stld_autosam_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation samples loaded from the selected eval split.",
    )
    train_stld_autosam_parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size for STLD AutoSAM training.",
    )
    train_stld_autosam_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count for STLD AutoSAM train/eval loaders.",
    )
    train_stld_autosam_parser.add_argument(
        "--train-repeat-factor",
        type=int,
        default=3,
        help="How many times the STLD train subset is repeated inside one AutoSAM epoch.",
    )
    train_stld_autosam_parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Adam learning rate for the upstream AutoSAM prompt generator.",
    )
    train_stld_autosam_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay for the upstream AutoSAM prompt generator.",
    )
    train_stld_autosam_parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs for the STLD AutoSAM prompt generator.",
    )
    train_stld_autosam_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.",
    )
    train_stld_autosam_parser.add_argument(
        "--train-augmentation-policy",
        choices=STLD_AUTOSAM_AUGMENTATION_POLICIES,
        default="autosam_dense_v1",
        help="Train-only paired image/mask augmentation policy for STLD AutoSAM.",
    )
    train_stld_autosam_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training and evaluation artifacts.",
    )
    add_save_visuals_args(
        train_stld_autosam_parser,
        "Save visualization panels during post-training evaluation.",
    )

    eval_stld_autosam_parser = subparsers.add_parser(
        "eval-stld-autosam",
        help="Evaluate a trained STLD AutoSAM checkpoint on one STLD split.",
    )
    eval_stld_autosam_parser.set_defaults(route="stld")
    eval_stld_autosam_parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the split-aware STLD root with ImageSets/Segmentation/{train,test}.txt plus images/ and labels/.",
    )
    eval_stld_autosam_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for STLD AutoSAM evaluation.",
    )
    eval_stld_autosam_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint produced by train-stld-autosam.",
    )
    eval_stld_autosam_parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="STLD split used for checkpoint evaluation.",
    )
    eval_stld_autosam_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated rows.",
    )
    eval_stld_autosam_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count for STLD AutoSAM evaluation.",
    )
    eval_stld_autosam_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.",
    )
    eval_stld_autosam_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_dataset_partition_args(eval_stld_autosam_parser)
    add_save_visuals_args(eval_stld_autosam_parser, "Save visualization panels during eval.")

    train_caid_autosam_parser = subparsers.add_parser(
        "train-caid-autosam",
        help="Train the upstream AutoSAM prompt-generator route on the prepared CAID split.",
    )
    train_caid_autosam_parser.set_defaults(route="caid")
    train_caid_autosam_parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the split-aware CAID root with ImageSets/Segmentation/{train,test}.txt plus images/ and labels/.",
    )
    train_caid_autosam_parser.add_argument("--device", default="cuda", help="Torch device used for CAID AutoSAM training and evaluation.")
    train_caid_autosam_parser.add_argument("--sam-checkpoint-path", default=DEFAULT_STLD_AUTOSAM_CHECKPOINT_PATH, help="Path to the official Meta SAM checkpoint used by AutoSAM.")
    train_caid_autosam_parser.add_argument("--sam-model-type", choices=("vit_h", "vit_l", "vit_b"), default="vit_h", help="SAM model type passed into the upstream AutoSAM SAM builder.")
    train_caid_autosam_parser.add_argument("--prompt-image-size", type=int, default=256, help="Square prompt-generator input size (Idim in the upstream AutoSAM code).")
    train_caid_autosam_parser.add_argument("--autosam-backbone-order", type=int, default=85, help="HarDNet order for the upstream AutoSAM prompt generator.")
    train_caid_autosam_parser.add_argument("--autosam-depth-wise", action="store_true", help="Enable the upstream AutoSAM depth-wise HarDNet variant.")
    train_caid_autosam_parser.add_argument("--train-split", choices=("train", "test"), default="train", help="CAID split used for optimization.")
    train_caid_autosam_parser.add_argument("--eval-split", choices=("train", "test"), default="test", help="CAID split used for post-training evaluation.")
    train_caid_autosam_parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on the number of training samples loaded from the selected train split.")
    train_caid_autosam_parser.add_argument("--train-subset-manifest", default=None, help="Optional JSON manifest selecting an explicit CAID train subset by image ID. Mutually exclusive with --train-limit.")
    train_caid_autosam_parser.add_argument("--eval-limit", type=int, default=None, help="Optional cap on the number of evaluation samples loaded from the selected eval split.")
    train_caid_autosam_parser.add_argument("--batch-size", type=int, default=3, help="Batch size for CAID AutoSAM training.")
    train_caid_autosam_parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count for CAID AutoSAM train/eval loaders.")
    train_caid_autosam_parser.add_argument("--train-repeat-factor", type=int, default=3, help="How many times the CAID train subset is repeated inside one AutoSAM epoch.")
    train_caid_autosam_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate for the upstream AutoSAM prompt generator.")
    train_caid_autosam_parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay for the upstream AutoSAM prompt generator.")
    train_caid_autosam_parser.add_argument("--num-epochs", type=int, default=200, help="Number of training epochs for the CAID AutoSAM prompt generator.")
    train_caid_autosam_parser.add_argument("--foreground-threshold", type=float, default=0.5, help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.")
    train_caid_autosam_parser.add_argument("--train-augmentation-policy", choices=STLD_AUTOSAM_AUGMENTATION_POLICIES, default="autosam_dense_v1", help="Train-only paired image/mask augmentation policy for CAID AutoSAM.")
    train_caid_autosam_parser.add_argument("--output-dir", default=None, help="Output directory for training and evaluation artifacts.")
    add_save_visuals_args(train_caid_autosam_parser, "Save visualization panels during post-training evaluation.")

    eval_caid_autosam_parser = subparsers.add_parser(
        "eval-caid-autosam",
        help="Evaluate a trained CAID AutoSAM checkpoint on one CAID split.",
    )
    eval_caid_autosam_parser.set_defaults(route="caid")
    eval_caid_autosam_parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to the split-aware CAID root with ImageSets/Segmentation/{train,test}.txt plus images/ and labels/.",
    )
    eval_caid_autosam_parser.add_argument("--device", default="cuda", help="Torch device used for CAID AutoSAM evaluation.")
    eval_caid_autosam_parser.add_argument("--checkpoint-path", required=True, help="Path to a checkpoint produced by train-caid-autosam.")
    eval_caid_autosam_parser.add_argument("--split", choices=("train", "test"), default="test", help="CAID split used for checkpoint evaluation.")
    eval_caid_autosam_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_caid_autosam_parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count for CAID AutoSAM evaluation.")
    eval_caid_autosam_parser.add_argument("--foreground-threshold", type=float, default=0.5, help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.")
    eval_caid_autosam_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    add_dataset_partition_args(eval_caid_autosam_parser)
    add_save_visuals_args(eval_caid_autosam_parser, "Save visualization panels during eval.")

    train_rwtd_autosam_parser = subparsers.add_parser(
        "train-rwtd-autosam",
        help="Train the upstream AutoSAM prompt-generator route on the curated RWTD split with texture_a as foreground.",
    )
    train_rwtd_autosam_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="RWTD dataset identifier or local curated alias.")
    train_rwtd_autosam_parser.add_argument("--device", default="cuda", help="Torch device used for RWTD AutoSAM training and evaluation.")
    train_rwtd_autosam_parser.add_argument("--sam-checkpoint-path", default=DEFAULT_STLD_AUTOSAM_CHECKPOINT_PATH, help="Path to the official Meta SAM checkpoint used by AutoSAM.")
    train_rwtd_autosam_parser.add_argument("--sam-model-type", choices=("vit_h", "vit_l", "vit_b"), default="vit_h", help="SAM model type passed into the upstream AutoSAM SAM builder.")
    train_rwtd_autosam_parser.add_argument("--prompt-image-size", type=int, default=256, help="Square prompt-generator input size (Idim in the upstream AutoSAM code).")
    train_rwtd_autosam_parser.add_argument("--autosam-backbone-order", type=int, default=85, help="HarDNet order for the upstream AutoSAM prompt generator.")
    train_rwtd_autosam_parser.add_argument("--autosam-depth-wise", action="store_true", help="Enable the upstream AutoSAM depth-wise HarDNet variant.")
    train_rwtd_autosam_parser.add_argument("--train-split", choices=("train", "test"), default="train", help="RWTD split used for optimization.")
    train_rwtd_autosam_parser.add_argument("--eval-split", choices=("train", "test"), default="test", help="RWTD split used for post-training evaluation.")
    train_rwtd_autosam_parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on the number of training samples loaded from the selected train split.")
    train_rwtd_autosam_parser.add_argument("--train-subset-manifest", default=None, help="Optional JSON manifest selecting an explicit RWTD train subset by crop id. Mutually exclusive with --train-limit.")
    train_rwtd_autosam_parser.add_argument("--eval-limit", type=int, default=None, help="Optional cap on the number of evaluation samples loaded from the selected eval split.")
    train_rwtd_autosam_parser.add_argument("--batch-size", type=int, default=3, help="Batch size for RWTD AutoSAM training.")
    train_rwtd_autosam_parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count for RWTD AutoSAM train/eval loaders.")
    train_rwtd_autosam_parser.add_argument("--train-repeat-factor", type=int, default=3, help="How many times the RWTD train subset is repeated inside one AutoSAM epoch.")
    train_rwtd_autosam_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate for the upstream AutoSAM prompt generator.")
    train_rwtd_autosam_parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay for the upstream AutoSAM prompt generator.")
    train_rwtd_autosam_parser.add_argument("--num-epochs", type=int, default=200, help="Number of training epochs for the RWTD AutoSAM prompt generator.")
    train_rwtd_autosam_parser.add_argument("--foreground-threshold", type=float, default=0.5, help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.")
    train_rwtd_autosam_parser.add_argument("--train-augmentation-policy", choices=STLD_AUTOSAM_AUGMENTATION_POLICIES, default="autosam_dense_v1", help="Train-only paired image/mask augmentation policy for RWTD AutoSAM.")
    train_rwtd_autosam_parser.add_argument("--output-dir", default=None, help="Output directory for training and evaluation artifacts.")
    add_save_visuals_args(train_rwtd_autosam_parser, "Save visualization panels during post-training evaluation.")

    eval_rwtd_autosam_parser = subparsers.add_parser(
        "eval-rwtd-autosam",
        help="Evaluate a trained RWTD AutoSAM checkpoint on one curated RWTD split.",
    )
    eval_rwtd_autosam_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="RWTD dataset identifier or local curated alias.")
    eval_rwtd_autosam_parser.add_argument("--device", default="cuda", help="Torch device used for RWTD AutoSAM evaluation.")
    eval_rwtd_autosam_parser.add_argument("--checkpoint-path", required=True, help="Path to a checkpoint produced by train-rwtd-autosam.")
    eval_rwtd_autosam_parser.add_argument("--split", choices=("train", "test"), default=DEFAULT_EVAL_SPLIT, help="RWTD split used for checkpoint evaluation.")
    eval_rwtd_autosam_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_rwtd_autosam_parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count for RWTD AutoSAM evaluation.")
    eval_rwtd_autosam_parser.add_argument("--foreground-threshold", type=float, default=0.5, help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.")
    eval_rwtd_autosam_parser.add_argument("--output-dir", default=None, help="Output directory for evaluation artifacts.")
    add_dataset_partition_args(eval_rwtd_autosam_parser)
    add_save_visuals_args(eval_rwtd_autosam_parser, "Save visualization panels during eval.")

    train_monuseg_autosam_faithful_parser = subparsers.add_parser(
        "train-monuseg-autosam-faithful",
        help="Train the MoNuSeg AutoSAM faithful-reproduction route on the official challenge split.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--dataset-name",
        default=MONUSEG_HF_DATASET_NAME,
        help="Hugging Face dataset name for the official-split MoNuSeg mirror.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache dir for MoNuSeg.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for MoNuSeg AutoSAM training and evaluation.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--sam-checkpoint-path",
        default=DEFAULT_STLD_AUTOSAM_CHECKPOINT_PATH,
        help="Path to the official Meta SAM checkpoint used by AutoSAM.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--sam-model-type",
        choices=("vit_h", "vit_l", "vit_b"),
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["sam_model_type"],
        help="SAM model type passed into the upstream AutoSAM SAM builder.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--reproduction-profile",
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["profile_name"],
        help="Named faithful-reproduction profile. Only the documented MoNuSeg faithful profile is supported.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--paper-faithful-monuseg",
        action="store_true",
        help="Force strict paper-faithful MoNuSeg parameters (512x512, Adam 3e-4, 200 epochs).",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--monuseg-frozen-sam-head",
        action="store_true",
        help="Run the Frozen SAM Head (f2+refine f1) variant instead of AutoSAM prompt-generator.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--autosam-freeze-prompt-bn",
        action="store_true",
        help="Freeze BatchNorm layers inside the prompt generator by forcing them into eval mode during training.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--prompt-image-size",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["prompt_image_size"],
        help="Square prompt-generator input size (Idim in the upstream AutoSAM code).",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--autosam-backbone-order",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["autosam_backbone_order"],
        help="HarDNet order for the upstream AutoSAM prompt generator.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--autosam-depth-wise",
        action="store_true",
        default=bool(MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["autosam_depth_wise"]),
        help="Enable the upstream AutoSAM depth-wise HarDNet variant.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--train-split",
        choices=("train", "test"),
        default="train",
        help="MoNuSeg split used for optimization.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--eval-split",
        choices=("train", "test"),
        default="test",
        help="MoNuSeg split used for final artifact-writing evaluation.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--selection-split",
        choices=("val",),
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["selection_split"],
        help="Split used for checkpoint selection during training. Locked to the deterministic repo-contract validation split.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training samples loaded from the selected train split.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--train-subset-manifest",
        default=None,
        help="Optional JSON manifest selecting an explicit MoNuSeg train subset by patient ID. Mutually exclusive with --train-limit.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation samples loaded from the selected selection split.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--batch-size",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["batch_size"],
        help="Mini-batch size for MoNuSeg AutoSAM training.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["gradient_accumulation_steps"],
        help="Number of gradient-accumulation mini-batches per optimizer step.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--drop-incomplete-accumulation",
        action="store_true",
        default=bool(MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["drop_incomplete_accumulation"]),
        help="Drop the final incomplete accumulation window at epoch end instead of stepping on it.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--num-workers",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["num_workers"],
        help="PyTorch DataLoader worker count for MoNuSeg AutoSAM train/eval loaders.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--train-repeat-factor",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["train_repeat_factor"],
        help="How many times the MoNuSeg train subset is repeated inside one AutoSAM epoch.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--learning-rate",
        type=float,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["learning_rate"],
        help="Adam learning rate for the upstream AutoSAM prompt generator.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--weight-decay",
        type=float,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["weight_decay"],
        help="Adam weight decay for the upstream AutoSAM prompt generator.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--num-epochs",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["num_epochs"],
        help="Number of training epochs for the MoNuSeg AutoSAM prompt generator.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["eval_every_epochs"],
        help="Checkpoint-selection evaluation cadence in epochs.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--selection-metric",
        choices=MONUSEG_AUTOSAM_SELECTION_METRICS,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["selection_metric"],
        help="Metric used for upstream-style checkpoint selection.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--selection-checkpoint-mode",
        choices=("best_eval", "final"),
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["selection_checkpoint_mode"],
        help="Which checkpoint view should be treated as the headline run artifact.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["foreground_threshold"],
        help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--train-augmentation-policy",
        choices=MONUSEG_AUTOSAM_AUGMENTATION_POLICIES,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["train_augmentation_policy"],
        help="Train-only paired image/mask augmentation policy for MoNuSeg AutoSAM.",
    )
    train_monuseg_autosam_faithful_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for training and evaluation artifacts.",
    )
    add_save_visuals_args(
        train_monuseg_autosam_faithful_parser,
        "Save visualization panels during post-training evaluation.",
    )

    eval_monuseg_autosam_faithful_parser = subparsers.add_parser(
        "eval-monuseg-autosam-faithful",
        help="Evaluate a trained MoNuSeg AutoSAM faithful checkpoint on one official split.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--dataset-name",
        default=MONUSEG_HF_DATASET_NAME,
        help="Hugging Face dataset name for the official-split MoNuSeg mirror.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache dir for MoNuSeg.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for MoNuSeg AutoSAM evaluation.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint produced by train-monuseg-autosam-faithful.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="MoNuSeg split used for checkpoint evaluation.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on evaluated rows.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--num-workers",
        type=int,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["num_workers"],
        help="PyTorch DataLoader worker count for MoNuSeg AutoSAM evaluation.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["foreground_threshold"],
        help="Probability threshold used to convert AutoSAM masks into a binary foreground map at eval time.",
    )
    eval_monuseg_autosam_faithful_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    add_dataset_partition_args(eval_monuseg_autosam_faithful_parser)
    add_save_visuals_args(eval_monuseg_autosam_faithful_parser, "Save visualization panels during eval.")

    train_glas_autosam_faithful_parser = subparsers.add_parser(
        "train-glas-autosam-faithful",
        help="Train the GlaS AutoSAM faithful-reproduction route on the official challenge split.",
    )
    add_glas_binary_root_arg(train_glas_autosam_faithful_parser)
    train_glas_autosam_faithful_parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for GlaS AutoSAM training and evaluation.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--sam-checkpoint-path",
        default=DEFAULT_STLD_AUTOSAM_CHECKPOINT_PATH,
        help="Path to the official Meta SAM checkpoint used by AutoSAM.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--sam-model-type",
        choices=("vit_h", "vit_l", "vit_b"),
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["sam_model_type"],
        help="SAM model type passed into the upstream AutoSAM SAM builder.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--reproduction-profile",
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["profile_name"],
        help="Named faithful-reproduction profile.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--freeze-prompt-bn",
        action="store_true",
        help="Freeze BatchNorm layers inside the prompt generator by forcing them into eval mode during training.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--prompt-image-size",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["prompt_image_size"],
        help="Square prompt-generator input size.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--autosam-backbone-order",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["autosam_backbone_order"],
        help="HarDNet order for the upstream AutoSAM prompt generator.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--autosam-depth-wise",
        action="store_true",
        default=bool(GLAS_AUTOSAM_REPRODUCTION_PROFILE["autosam_depth_wise"]),
        help="Enable the upstream AutoSAM depth-wise HarDNet variant.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--batch-size",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["batch_size"],
        help="Physical batch size.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["gradient_accumulation_steps"],
        help="Number of mini-batches accumulated per optimizer step.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--learning-rate",
        type=float,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["learning_rate"],
        help="Adam learning rate.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--weight-decay",
        type=float,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["weight_decay"],
        help="Adam weight decay.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--num-epochs",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["num_epochs"],
        help="Number of training epochs.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["eval_every_epochs"],
        help="Evaluation cadence.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--train-repeat_factor",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["train_repeat_factor"],
        help="Dataset repeat factor.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--num-workers",
        type=int,
        default=GLAS_AUTOSAM_REPRODUCTION_PROFILE["num_workers"],
        help="DataLoader worker count.",
    )
    train_glas_autosam_faithful_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory.",
    )

    eval_glas_autosam_faithful_parser = subparsers.add_parser(
        "eval-glas-autosam-faithful",
        help="Evaluate a trained GlaS AutoSAM faithful checkpoint.",
    )
    add_glas_binary_root_arg(eval_glas_autosam_faithful_parser)
    eval_glas_autosam_faithful_parser.add_argument(
        "--checkpoint-path",
        required=True,
    )
    eval_glas_autosam_faithful_parser.add_argument(
        "--device",
        default="cuda",
    )
    eval_glas_autosam_faithful_parser.add_argument(
        "--output-dir",
        default=None,
    )

    predict_architexture_binary_parser = subparsers.add_parser(
        "predict-architexture-binary",
        help="Run the pooled coarse-only SAM-feature experiment on one local ArchiTexture CAID/STLD sample.",
    )
    add_architexture_binary_args(predict_architexture_binary_parser)
    add_sam3_auto_model_args(predict_architexture_binary_parser)
    predict_architexture_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the selected route.",
    )
    predict_architexture_binary_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Natural-sorted row index within the resolved benchmark root.",
    )
    predict_architexture_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prediction artifacts.",
    )
    add_save_visuals_args(
        predict_architexture_binary_parser,
        "Save prediction visualization panels.",
    )

    eval_architexture_binary_parser = subparsers.add_parser(
        "eval-architexture-binary",
        help="Evaluate the pooled coarse-only SAM-feature experiment on a local ArchiTexture CAID/STLD benchmark.",
    )
    add_architexture_binary_args(eval_architexture_binary_parser)
    add_sam3_auto_model_args(eval_architexture_binary_parser)
    eval_architexture_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the selected route.",
    )
    eval_architexture_binary_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_architexture_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    eval_architexture_binary_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_architexture_binary_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_architexture_binary_parser.add_argument(
        "--wandb-project",
        default="architexture-binary-sam3",
        help="Weights & Biases project name.",
    )
    eval_architexture_binary_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_architexture_binary_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_dataset_partition_args(eval_architexture_binary_parser)
    add_save_visuals_args(eval_architexture_binary_parser, "Save visualization panels during eval.")

    predict_detexture_binary_parser = subparsers.add_parser(
        "predict-detexture-binary",
        help="Run one registered cross-dataset current-method SAM-feature experiment on one local DeTexture ADE20K crop.",
    )
    add_detexture_binary_args(predict_detexture_binary_parser)
    add_sam3_auto_model_args(predict_detexture_binary_parser)
    predict_detexture_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the local DeTexture ADE20K assets.",
    )
    predict_detexture_binary_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Natural-sorted row index within the resolved local DeTexture ADE20K root.",
    )
    predict_detexture_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prediction artifacts.",
    )
    add_save_visuals_args(
        predict_detexture_binary_parser,
        "Save prediction visualization panels.",
    )

    eval_detexture_binary_parser = subparsers.add_parser(
        "eval-detexture-binary",
        help="Evaluate one registered cross-dataset current-method experiment on the local DeTexture ADE20K assets.",
    )
    add_detexture_binary_args(eval_detexture_binary_parser)
    add_sam3_auto_model_args(eval_detexture_binary_parser)
    eval_detexture_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the local DeTexture ADE20K assets.",
    )
    eval_detexture_binary_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_detexture_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    eval_detexture_binary_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_detexture_binary_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_detexture_binary_parser.add_argument(
        "--wandb-project",
        default="detexture-binary-sam3",
        help="Weights & Biases project name.",
    )
    eval_detexture_binary_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_detexture_binary_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_dataset_partition_args(eval_detexture_binary_parser)
    add_save_visuals_args(eval_detexture_binary_parser, "Save visualization panels during eval.")

    predict_detexture_multi_parser = subparsers.add_parser(
        "predict-detexture-multi",
        help="Run one prompt-free multi-region pooled-coarsest SAM-feature experiment on one local DeTexture crop.",
    )
    add_detexture_multi_args(predict_detexture_multi_parser)
    add_sam3_auto_model_args(predict_detexture_multi_parser)
    predict_detexture_multi_parser.add_argument(
        "--variant",
        choices=DETEXTURE_MULTI_SUPPORTED_VARIANTS,
        default=DETEXTURE_MULTI_DEFAULT_VARIANT,
        help="Prompt-free multi-region DeTexture experiment to run.",
    )
    predict_detexture_multi_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Natural-sorted row index within the resolved local DeTexture multi root.",
    )
    add_deepdpm_args(predict_detexture_multi_parser)
    add_hdbscan_args(predict_detexture_multi_parser)
    predict_detexture_multi_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prediction artifacts.",
    )
    add_save_visuals_args(
        predict_detexture_multi_parser,
        "Save prediction visualization panels.",
    )

    eval_detexture_multi_parser = subparsers.add_parser(
        "eval-detexture-multi",
        help="Evaluate one prompt-free multi-region DeTexture experiment on the local DeTexture ADE20K assets.",
    )
    add_detexture_multi_args(eval_detexture_multi_parser)
    add_sam3_auto_model_args(eval_detexture_multi_parser)
    eval_detexture_multi_parser.add_argument(
        "--variant",
        choices=DETEXTURE_MULTI_SUPPORTED_VARIANTS,
        default=DETEXTURE_MULTI_DEFAULT_VARIANT,
        help="Prompt-free multi-region DeTexture experiment to run.",
    )
    add_deepdpm_args(eval_detexture_multi_parser)
    add_hdbscan_args(eval_detexture_multi_parser)
    eval_detexture_multi_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_detexture_multi_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    eval_detexture_multi_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_detexture_multi_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_detexture_multi_parser.add_argument(
        "--wandb-project",
        default="detexture-multi-sam3",
        help="Weights & Biases project name.",
    )
    eval_detexture_multi_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_detexture_multi_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_dataset_partition_args(eval_detexture_multi_parser)
    add_save_visuals_args(eval_detexture_multi_parser, "Save visualization panels during eval.")

    predict_cstd_binary_parser = subparsers.add_parser(
        "predict-cstd-binary",
        help="Run one registered cross-dataset current-method SAM-feature experiment on one local CSTD sample.",
    )
    add_cstd_binary_args(predict_cstd_binary_parser)
    add_sam3_auto_model_args(predict_cstd_binary_parser)
    predict_cstd_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the local CSTD assets.",
    )
    predict_cstd_binary_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Natural-sorted row index within the resolved local CSTD root.",
    )
    predict_cstd_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prediction artifacts.",
    )
    add_save_visuals_args(
        predict_cstd_binary_parser,
        "Save prediction visualization panels.",
    )

    eval_cstd_binary_parser = subparsers.add_parser(
        "eval-cstd-binary",
        help="Evaluate one registered cross-dataset current-method experiment on the local CSTD assets.",
    )
    add_cstd_binary_args(eval_cstd_binary_parser)
    add_sam3_auto_model_args(eval_cstd_binary_parser)
    eval_cstd_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the local CSTD assets.",
    )
    eval_cstd_binary_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_cstd_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    eval_cstd_binary_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_cstd_binary_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_cstd_binary_parser.add_argument(
        "--wandb-project",
        default="cstd-binary-sam3",
        help="Weights & Biases project name.",
    )
    eval_cstd_binary_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_cstd_binary_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB previews.",
    )
    add_dataset_partition_args(eval_cstd_binary_parser)
    add_save_visuals_args(eval_cstd_binary_parser, "Save visualization panels during eval.")


    predict_glas_binary_parser = subparsers.add_parser(
        "predict-glas-binary",
        help="Run one registered cross-dataset current-method SAM-feature experiment on one local GlaS sample.",
    )
    add_glas_binary_args(predict_glas_binary_parser)
    add_sam3_auto_model_args(predict_glas_binary_parser)
    predict_glas_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the local GlaS assets.",
    )
    predict_glas_binary_parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Natural-sorted row index within the resolved local GlaS root and requested split.",
    )
    predict_glas_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prediction artifacts.",
    )
    add_save_visuals_args(
        predict_glas_binary_parser,
        "Save prediction visualization panels.",
    )

    eval_glas_binary_parser = subparsers.add_parser(
        "eval-glas-binary",
        help="Evaluate one registered cross-dataset current-method experiment on the local GlaS assets.",
    )
    add_glas_binary_args(eval_glas_binary_parser)
    add_sam3_auto_model_args(eval_glas_binary_parser)
    eval_glas_binary_parser.add_argument(
        "--variant",
        choices=CROSS_DATASET_EXPERIMENT_VARIANTS,
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
        help="Cross-dataset SAM-3 feature experiment to run on the local GlaS assets.",
    )
    eval_glas_binary_parser.add_argument("--limit", type=int, default=None, help="Optional cap on evaluated rows.")
    eval_glas_binary_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    eval_glas_binary_parser.add_argument(
        "--failure-policy",
        choices=("abort", "skip"),
        default="abort",
        help="Whether to abort on the first failed sample or skip failures.",
    )
    eval_glas_binary_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    eval_glas_binary_parser.add_argument(
        "--wandb-project",
        default="glas-binary-sam3",
        help="Weights & Biases project name.",
    )
    eval_glas_binary_parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional explicit WandB run name.",
    )
    eval_glas_binary_parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Preview logging cadence for WandB images.",
    )
    add_dataset_partition_args(eval_glas_binary_parser)
    add_save_visuals_args(eval_glas_binary_parser, "Save visualization panels during eval.")

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="RWTD dataset identifier. Defaults to the curated repo/Hugging Face RWTD bundle; legacy public HF RWTD is still accepted explicitly.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default=DEFAULT_EVAL_SPLIT,
        help="Dataset view to use. 'all' concatenates the public train and test splits.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory. Defaults to a writable repo-local cache.",
    )


def add_kaust256_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rwtd-root",
        required=True,
        help="Path to the official TextureSAM Kaust256 dataset root or the parent directory containing `Kaust256/`.",
    )


def add_architexture_binary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--route",
        choices=("stld", "caid"),
        required=True,
        help="ArchiTexture benchmark route to run.",
    )
    parser.add_argument(
        "--benchmark-root",
        required=True,
        help=(
            "Path to the local ArchiTexture benchmark root or its documented experiment root. "
            "For STLD this resolves `<root>/benchmark`; for CAID this resolves `<root>/benchmarks/caid_test`."
        ),
    )


def add_detexture_binary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Path to the local DeTexture ADE20K root. The resolver accepts either the dataset root containing "
            "`assets/crops` and `assets/masks`, or the `assets/` directory itself."
        ),
    )


def add_detexture_multi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Path to the local multi-region DeTexture root. The resolver accepts either the parent directory "
            "containing `detecture_data/images` and `detecture_data/masks`, or `detecture_data/` itself."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("all", "training", "validation"),
        default="all",
        help="Subset of the local multi-region DeTexture assets to evaluate.",
    )


def add_deepdpm_args(parser: argparse.ArgumentParser) -> None:
    """Add optional DeepDPM hyperparameter overrides for the multi-region DeTexture route."""

    parser.add_argument("--deepdpm-init-clusters", type=int, default=None, help="Optional initial cluster count for the DeepDPM-style head.")
    parser.add_argument("--deepdpm-min-clusters", type=int, default=None, help="Optional minimum allowed cluster count for the DeepDPM-style head.")
    parser.add_argument("--deepdpm-max-clusters", type=int, default=None, help="Optional maximum allowed cluster count for the DeepDPM-style head.")
    parser.add_argument("--deepdpm-hidden-dim", type=int, default=None, help="Optional hidden dimension for the DeepDPM-style encoder MLP.")
    parser.add_argument("--deepdpm-embedding-dim", type=int, default=None, help="Optional embedding dimension for the DeepDPM-style clustering space.")
    parser.add_argument("--deepdpm-outer-iterations", type=int, default=None, help="Optional number of DeepDPM-style split/merge outer iterations per sample.")
    parser.add_argument("--deepdpm-inner-epochs", type=int, default=None, help="Optional number of inner optimization epochs per outer iteration.")
    parser.add_argument("--deepdpm-learning-rate", type=float, default=None, help="Optional learning rate for the DeepDPM-style clustering head.")
    parser.add_argument("--deepdpm-weight-decay", type=float, default=None, help="Optional weight decay for the DeepDPM-style clustering head optimizer.")
    parser.add_argument("--deepdpm-split-dispersion-threshold", type=float, default=None, help="Optional dispersion threshold that triggers deterministic cluster splits.")
    parser.add_argument("--deepdpm-merge-similarity-threshold", type=float, default=None, help="Optional cosine-similarity threshold that triggers deterministic cluster merges.")


def add_hdbscan_args(parser: argparse.ArgumentParser) -> None:
    """Add optional HDBSCAN hyperparameter overrides for the multi-region DeTexture route."""

    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=None,
        help="Optional minimum HDBSCAN cluster size on pooled frozen SAM feature vectors.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="Optional HDBSCAN min_samples override on pooled frozen SAM feature vectors.",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=None,
        help="Optional HDBSCAN cluster-selection epsilon on pooled frozen SAM feature vectors.",
    )


def add_cstd_binary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the local CSTD root containing `images/`, `regions/`, and `edges/`.",
    )


def add_glas_binary_args(parser: argparse.ArgumentParser) -> None:
    add_glas_binary_root_arg(parser)
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
        help="Subset of the local GlaS assets to evaluate.",
    )


def add_glas_binary_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Path to the local GlaS root. The resolver accepts either the flat directory containing "
            "`train_*`, `testA_*`, `testB_*`, and `*_anno.bmp` files, or an extracted Warwick archive root "
            "that contains those files somewhere below it."
        ),
    )


def add_monuseg_binary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-name",
        default=MONUSEG_HF_DATASET_NAME,
        help="Hugging Face dataset name used for MoNuSeg loading.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory used for MoNuSeg.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
        help="Official MoNuSeg split view to evaluate. Train excludes the 7 extra `tissue == 0` Hugging Face rows.",
    )


def add_yknd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        default="datasets/YKND",
        help="Local root directory for the YKND dataset.",
    )


def add_save_visuals_args(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add default-on visual output flags with an explicit opt-out."""

    parser.add_argument(
        "--save-visuals",
        dest="save_visuals",
        action="store_true",
        default=True,
        help=help_text,
    )
    parser.add_argument(
        "--no-save-visuals",
        dest="save_visuals",
        action="store_false",
        help="Disable visual output generation.",
    )
    parser.add_argument(
        "--save-pooled-feature-pca-overlay",
        dest="save_pooled_feature_pca_overlay",
        action="store_true",
        default=False,
        help=(
            "Save the pooled coarsest-feature PCA overlay on the input image, using the same "
            "PCA-RGB projection as the coarse-feature figure export script."
        ),
    )


def add_dataset_partition_args(parser: argparse.ArgumentParser) -> None:
    """Add deterministic `K/N` dataset partition selection for eval commands."""

    parser.add_argument(
        "--dataset-partition",
        default=None,
        help=(
            "Optional deterministic dataset partition spec of the form 'K/N'. "
            "Partitioning follows the dataset's natural order, then --limit is applied within that partition."
        ),
    )


def add_coarse_vs_fine_stage2_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset-source flags for the Stage-2 coarse-vs-fine probe study."""

    parser.add_argument(
        "--dataset-source",
        choices=("rwtd", "architexture_binary", "cstd_binary"),
        default="rwtd",
        help="Dataset source used by the Stage-2 probe train/eval commands.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="RWTD dataset identifier used when --dataset-source rwtd. Defaults to the curated repo/Hugging Face RWTD bundle.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory used when --dataset-source rwtd.",
    )
    parser.add_argument(
        "--route",
        choices=("stld", "caid"),
        default=None,
        help="ArchiTexture route used when --dataset-source architexture_binary.",
    )
    parser.add_argument(
        "--benchmark-root",
        default=None,
        help=(
            "Local ArchiTexture root used when --dataset-source architexture_binary. For Stage-2 training/eval this must point at an official split-aware root, not the flat benchmark-only copy."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Local CSTD root used when --dataset-source cstd_binary. For Stage-2 training/eval this must point at an official split-aware root, not the flat benchmark-only copy."
        ),
    )



def add_coarse_vs_fine_stage2_probe_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared tiny-probe hyperparameters for the Stage-2 coarse-vs-fine study."""

    parser.add_argument("--projection-dim", type=int, default=64, help="Per-level 1x1 projection output dimension d.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Final per-pixel embedding dimension d_emb.")
    parser.add_argument(
        "--pairs-per-image",
        type=int,
        default=512,
        help="Number of sampled same/different affinity pairs per image on the coarsest grid.",
    )
    parser.add_argument(
        "--affinity-temperature",
        type=float,
        default=0.10,
        help="Temperature divisor applied to cosine affinity logits before BCE.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate for the tiny probe.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay for the tiny probe.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of probe-training epochs.")
    parser.add_argument(
        "--kmeans-metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Deterministic k-means metric used to cluster learned embeddings.",
    )
    parser.add_argument(
        "--kmeans-num-clusters",
        type=int,
        default=2,
        help="Number of clusters used during evaluation. Stage 2 fixes this to 2.",
    )
    parser.add_argument(
        "--kmeans-max-iterations",
        type=int,
        default=25,
        help="Maximum deterministic k-means iterations on learned embeddings.",
    )
    parser.add_argument(
        "--kmeans-convergence-tolerance",
        type=float,
        default=1e-4,
        help="Deterministic k-means centroid-shift tolerance on learned embeddings.",
    )


def add_coarse_vs_fine_linear_probe_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared few-shot linear-probe hyperparameters."""

    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate for the linear probe.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay for the linear probe.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of linear-probe training epochs.")
    parser.set_defaults(post_concat_normalization=True)
    parser.add_argument(
        "--no-post-concat-normalization",
        dest="post_concat_normalization",
        action="store_false",
        help="Disable post-concatenation per-pixel L2 normalization before the linear 1x1 classifier.",
    )


def add_residual_head_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    include_training_losses: bool,
    include_eval_alpha_override: bool,
) -> None:
    """Add explicit control flags for coarse-plus-fine residual frozen-mask-head variants."""

    parser.add_argument(
        "--residual-gate-mode",
        choices=RESIDUAL_HEAD_GATE_MODES,
        default="none",
        help="Explicit residual gating mode for coarse-plus-fine residual heads. `none` keeps the current unconditional residual path; `hard_uncertainty` and `soft_uncertainty` gate residual logits using coarse-logit uncertainty.",
    )
    parser.add_argument(
        "--residual-gate-threshold",
        type=float,
        default=float(RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD),
        help="Uncertainty threshold used by the residual gate modes. Pixels with coarse probabilities within this distance of 0.5 are considered uncertain.",
    )
    parser.add_argument(
        "--attention-hidden-dim",
        type=int,
        default=int(FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"]),
        help="Hidden width used by the learned attention refine heads (`fpn_2_plus_fpn_1_attn_refine`, `fpn_2_plus_fpn_1_cross_attn_refine`).",
    )
    parser.add_argument(
        "--cross-attn-query-stride",
        type=int,
        default=int(FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"]),
        help="Spatial stride used to downsample coarse-guided cross-attention queries before they attend into full `fpn_1` keys/values in `fpn_2_plus_fpn_1_cross_attn_refine`.",
    )
    if include_eval_alpha_override:
        parser.add_argument(
            "--residual-alpha-override",
            type=float,
            default=None,
            help="Evaluation-only override for the learned residual scale in coarse-plus-fine residual heads.",
        )
    if include_training_losses:
        parser.add_argument(
            "--coarse-loss-weight",
            type=float,
            default=0.0,
            help="Optional auxiliary BCE+Dice weight applied to coarse logits in residual-head variants.",
        )
        parser.add_argument(
            "--residual-l1-weight",
            type=float,
            default=0.0,
            help="Optional L1 penalty weight applied to residual logits in residual-head variants.",
        )
        parser.add_argument(
            "--attention-sparsity-weight",
            type=float,
            default=0.0,
            help="Optional L1-style sparsity penalty weight applied to learned attention maps in attention-refine residual-head variants.",
        )
    parser.add_argument(
        "--joker-disable-fpn0",
        action="store_true",
        help="Disable the `fpn_0` evidence bank in `fpn_2_plus_multibank_null_refine`.",
    )
    parser.add_argument(
        "--joker-disable-fpn1",
        action="store_true",
        help="Disable the `fpn_1` evidence bank in `fpn_2_plus_multibank_null_refine`.",
    )
    parser.add_argument(
        "--joker-disable-null-token",
        action="store_true",
        help="Remove the learned null token from `fpn_2_plus_multibank_null_refine`.",
    )
    parser.add_argument(
        "--joker-use-learned-gate",
        action="store_true",
        help="Enable one learned residual gate on top of `fpn_2_plus_multibank_null_refine`. This is reserved for the late gate ablation, not the default joker contract.",
    )
    parser.add_argument(
        "--joker-zero-init-residual-scale",
        action="store_true",
        help="Explicitly zero-initialize the joker residual fusion scale so training starts from a coarse-only prediction.",
    )
    parser.add_argument(
        "--joker-zero-init-attn-qkv",
        action="store_true",
        help="Zero-initialize the joker attention query/key/value projection layers. This only applies to `fpn_2_plus_multibank_null_refine`.",
    )
    parser.add_argument(
        "--joker-residual-warmup-epochs",
        type=int,
        default=0,
        help="Clamp the joker residual contribution to zero for the first N epochs, then enable full residual strength immediately afterward.",
    )
    parser.add_argument(
        "--joker-residual-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp the joker residual contribution from zero to full strength over the first N epochs.",
    )


def add_eval_loader_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation dataloader tuning flags for the prompt-conditioned RWTD path."""

    cpu_count = os.cpu_count() or 1
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PROMPT_EVAL_DEFAULT_BATCH_SIZE,
        help="Evaluation batch size for text prompting. Oracle-point prompting still runs per sample within each batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, cpu_count),
        help="Number of dataloader worker processes used to decode/eagerly prefetch evaluation samples.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=PROMPT_EVAL_DEFAULT_PREFETCH_FACTOR,
        help="Per-worker dataloader prefetch factor used when num_workers > 0.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="SAM 3 model identifier.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated model access.",
    )
    parser.add_argument(
        "--official-checkpoint-path",
        default=None,
        help="Optional path to a local official Meta SAM 3 checkpoint.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for keeping predicted instances.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold for converting predicted logits into binary masks.",
    )
    parser.add_argument(
        "--boundary-tolerance-px",
        type=int,
        default=2,
        help="Boundary metric tolerance in pixels.",
    )


def add_sam2_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_SAM2_MODEL_ID, help="SAM-2 model identifier.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for model download/authenticated cache access.",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="Override the automatic mask generator point batch size. Larger values are faster on large GPUs.",
    )
    parser.add_argument(
        "--fast-cuda",
        action="store_true",
        help="Enable CUDA autocast with bfloat16 and TF32 for faster SAM-2 inference on Ampere+ GPUs.",
    )
    parser.add_argument(
        "--compile-image-encoder",
        action="store_true",
        help="Enable torch.compile for the SAM-2 image encoder. Higher startup cost, lower steady-state latency.",
    )


def add_sam3_auto_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_SAM3_AUTO_MODEL_ID, help="SAM-3 model identifier.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated model access.",
    )
    parser.add_argument(
        "--official-checkpoint-path",
        default=None,
        help="Optional local official Meta SAM 3 checkpoint used by the SAM-feature refinement paths.",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the RWTD SAM 3 CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    set_global_seed(args.seed)

    if getattr(args, "log_every", 1) < 1:
        parser.error("--log-every must be at least 1.")
    if getattr(args, "boundary_tolerance_px", 0) < 0:
        parser.error("--boundary-tolerance-px must be non-negative.")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive when provided.")
    if getattr(args, "dataset_partition", None) is not None:
        try:
            parse_dataset_partition_spec(args.dataset_partition)
        except ValueError as exc:
            parser.error(str(exc))
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be at least 1.")
    if getattr(args, "gradient_accumulation_steps", 1) < 1:
        parser.error("--gradient-accumulation-steps must be at least 1.")
    if getattr(args, "num_workers", 0) < 0:
        parser.error("--num-workers must be non-negative.")
    if getattr(args, "prefetch_factor", 1) < 1:
        parser.error("--prefetch-factor must be at least 1.")
    if getattr(args, "deepdpm_init_clusters", None) is not None and args.deepdpm_init_clusters < 1:
        parser.error("--deepdpm-init-clusters must be at least 1.")
    if getattr(args, "deepdpm_min_clusters", None) is not None and args.deepdpm_min_clusters < 1:
        parser.error("--deepdpm-min-clusters must be at least 1.")
    if getattr(args, "deepdpm_max_clusters", None) is not None and args.deepdpm_max_clusters < 1:
        parser.error("--deepdpm-max-clusters must be at least 1.")
    if getattr(args, "deepdpm_hidden_dim", None) is not None and args.deepdpm_hidden_dim < 1:
        parser.error("--deepdpm-hidden-dim must be at least 1.")
    if getattr(args, "deepdpm_embedding_dim", None) is not None and args.deepdpm_embedding_dim < 1:
        parser.error("--deepdpm-embedding-dim must be at least 1.")
    if getattr(args, "deepdpm_outer_iterations", None) is not None and args.deepdpm_outer_iterations < 1:
        parser.error("--deepdpm-outer-iterations must be at least 1.")
    if getattr(args, "deepdpm_inner_epochs", None) is not None and args.deepdpm_inner_epochs < 1:
        parser.error("--deepdpm-inner-epochs must be at least 1.")
    if getattr(args, "deepdpm_learning_rate", None) is not None and args.deepdpm_learning_rate <= 0:
        parser.error("--deepdpm-learning-rate must be positive.")
    if getattr(args, "deepdpm_weight_decay", None) is not None and args.deepdpm_weight_decay < 0:
        parser.error("--deepdpm-weight-decay must be non-negative.")
    if getattr(args, "deepdpm_split_dispersion_threshold", None) is not None and args.deepdpm_split_dispersion_threshold <= 0:
        parser.error("--deepdpm-split-dispersion-threshold must be positive.")
    if getattr(args, "deepdpm_merge_similarity_threshold", None) is not None and not (-1.0 <= args.deepdpm_merge_similarity_threshold <= 1.0):
        parser.error("--deepdpm-merge-similarity-threshold must be between -1 and 1.")
    if getattr(args, "hdbscan_min_cluster_size", None) is not None and args.hdbscan_min_cluster_size < 2:
        parser.error("--hdbscan-min-cluster-size must be at least 2.")
    if getattr(args, "hdbscan_min_samples", None) is not None and args.hdbscan_min_samples < 1:
        parser.error("--hdbscan-min-samples must be at least 1.")
    if getattr(args, "hdbscan_cluster_selection_epsilon", None) is not None and args.hdbscan_cluster_selection_epsilon < 0:
        parser.error("--hdbscan-cluster-selection-epsilon must be non-negative.")

    if getattr(args, "train_limit", None) is not None and args.train_limit < 1:
        parser.error("--train-limit must be positive when provided.")
    if getattr(args, "num_train_samples", None) is not None and args.num_train_samples < 1:
        parser.error("--num-train-samples must be positive when provided.")
    if getattr(args, "eval_limit", None) is not None and args.eval_limit < 1:
        parser.error("--eval-limit must be positive when provided.")
    if getattr(args, "projection_dim", None) is not None and args.projection_dim < 1:
        parser.error("--projection-dim must be at least 1.")
    if getattr(args, "decoder_dim", None) is not None and args.decoder_dim < 1:
        parser.error("--decoder-dim must be at least 1.")
    if getattr(args, "memory_token_count", None) is not None and args.memory_token_count < 1:
        parser.error("--memory-token-count must be at least 1.")
    if getattr(args, "attention_heads", None) is not None and args.attention_heads < 1:
        parser.error("--attention-heads must be at least 1.")
    if getattr(args, "attention_blocks", None) is not None and args.attention_blocks < 1:
        parser.error("--attention-blocks must be at least 1.")
    if getattr(args, "embedding_dim", None) is not None and args.embedding_dim < 1:
        parser.error("--embedding-dim must be at least 1.")
    if getattr(args, "pairs_per_image", None) is not None and args.pairs_per_image < 2:
        parser.error("--pairs-per-image must be at least 2.")
    if getattr(args, "affinity_temperature", None) is not None and args.affinity_temperature <= 0:
        parser.error("--affinity-temperature must be positive.")
    if getattr(args, "learning_rate", None) is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if getattr(args, "learning_rate", None) is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if getattr(args, "weight_decay", None) is not None and args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative.")
    if getattr(args, "num_epochs", None) is not None and args.num_epochs < 1:
        parser.error("--num-epochs must be at least 1.")
    if getattr(args, "eval_every_epochs", None) is not None and args.eval_every_epochs < 1:
        parser.error("--eval-every-epochs must be positive when provided.")
    if getattr(args, "bce_weight", None) is not None and args.bce_weight < 0:
        parser.error("--bce-weight must be non-negative.")
    if getattr(args, "dice_weight", None) is not None and args.dice_weight < 0:
        parser.error("--dice-weight must be non-negative.")
    if getattr(args, "boundary_weight", None) is not None and args.boundary_weight < 0:
        parser.error("--boundary-weight must be non-negative.")
    if getattr(args, "foreground_threshold", None) is not None and not (0.0 <= args.foreground_threshold <= 1.0):
        parser.error("--foreground-threshold must satisfy 0 <= threshold <= 1.")
    resize_height = getattr(args, "resize_height", None)
    resize_width = getattr(args, "resize_width", None)
    if (resize_height is None) != (resize_width is None):
        parser.error("--resize-height and --resize-width must be provided together.")
    if resize_height is not None and resize_height < 1:
        parser.error("--resize-height must be positive.")
    if resize_width is not None and resize_width < 1:
        parser.error("--resize-width must be positive.")
    backbone_preprocessing_mode = getattr(args, "backbone_preprocessing_mode", None)
    if (
        backbone_preprocessing_mode is not None
        and backbone_preprocessing_mode != "native_resolution"
        and (resize_height is not None or resize_width is not None)
    ):
        parser.error("--backbone-preprocessing-mode autosam_style_1024 cannot be combined with --resize-height/--resize-width.")
    if getattr(args, "decoder_dim", None) is not None and args.command in {
        "train-glas-frozen-sam-mask-head",
        "train-monuseg-frozen-sam-mask-head",
        "train-stld-frozen-sam-mask-head",
        "train-caid-frozen-sam-mask-head",
        "train-rwtd-frozen-sam-mask-head",
    }:
        if args.decoder_dim % 8 != 0:
            parser.error("--decoder-dim must be divisible by 8 for the fixed GroupNorm setting in the frozen mask head.")
    if getattr(args, "variant", None) in {"fpn_2_memory_attn", "fpn_2_plus_fpn_1_memory_attn"}:
        if getattr(args, "projection_dim", None) is not None and getattr(args, "attention_heads", None) is not None:
            if args.projection_dim % args.attention_heads != 0:
                parser.error("--projection-dim must be divisible by --attention-heads for memory-attention variants.")
    if getattr(args, "kmeans_num_clusters", None) is not None and args.kmeans_num_clusters != 2:
        parser.error("--kmeans-num-clusters must be exactly 2 for the coarse-vs-fine Stage-2 study.")

    if args.command in {
        "train-coarse-vs-fine-sam-probe",
        "eval-coarse-vs-fine-sam-probe",
        "train-coarse-vs-fine-linear-probe",
        "eval-coarse-vs-fine-linear-probe",
    }:
        dataset_source = getattr(args, "dataset_source", "rwtd")
        if dataset_source == "architexture_binary":
            if getattr(args, "route", None) is None:
                parser.error("--route is required when --dataset-source architexture_binary.")
            if getattr(args, "benchmark_root", None) is None:
                parser.error("--benchmark-root is required when --dataset-source architexture_binary.")
            if getattr(args, "dataset_root", None) is not None:
                parser.error("--dataset-root is only valid when --dataset-source cstd_binary.")
        elif dataset_source == "cstd_binary":
            if getattr(args, "dataset_root", None) is None:
                parser.error("--dataset-root is required when --dataset-source cstd_binary.")
            if getattr(args, "benchmark_root", None) is not None:
                parser.error("--benchmark-root is only valid when --dataset-source architexture_binary.")
            if getattr(args, "route", None) is not None:
                parser.error("--route is only valid when --dataset-source architexture_binary.")
        else:
            if getattr(args, "benchmark_root", None) is not None:
                parser.error("--benchmark-root is only valid when --dataset-source architexture_binary.")
            if getattr(args, "route", None) is not None:
                parser.error("--route is only valid when --dataset-source architexture_binary.")
            if getattr(args, "dataset_root", None) is not None:
                parser.error("--dataset-root is only valid when --dataset-source cstd_binary.")
        if getattr(args, "num_train_samples", None) is not None:
            if getattr(args, "train_limit", None) is not None and args.train_limit != args.num_train_samples:
                parser.error("--num-train-samples and --train-limit must match when both are provided.")
            args.train_limit = args.num_train_samples
    if getattr(args, "kmeans_max_iterations", None) is not None and args.kmeans_max_iterations < 1:
        parser.error("--kmeans-max-iterations must be at least 1.")
    if getattr(args, "kmeans_convergence_tolerance", None) is not None and args.kmeans_convergence_tolerance <= 0:
        parser.error("--kmeans-convergence-tolerance must be positive.")
    if getattr(args, "eval_every_epochs", None) is not None and args.eval_every_epochs < 1:
        parser.error("--eval-every-epochs must be at least 1.")
    if args.command in {
        "train-glas-frozen-sam-mask-head",
        "train-monuseg-frozen-sam-mask-head",
    }:
        selection_mode = getattr(args, "selection_checkpoint_mode", "final")
        eval_every_epochs = getattr(args, "eval_every_epochs", None)
        validation_holdout_count = int(getattr(args, "validation_holdout_count", 0)) if hasattr(args, "validation_holdout_count") else 0
        selection_split = getattr(args, "selection_split", None)
        if selection_mode == "best_eval" and eval_every_epochs is None:
            parser.error("--selection-checkpoint-mode best_eval requires --eval-every-epochs so a best checkpoint can be materialized.")
        if selection_mode == "best_eval" and selection_split == "val" and validation_holdout_count < 1:
            parser.error("--selection-split val requires --validation-holdout-count > 0.")
    if args.command in {
        "train-glas-frozen-sam-mask-head",
        "eval-glas-frozen-sam-mask-head",
        "train-monuseg-frozen-sam-mask-head",
        "eval-monuseg-frozen-sam-mask-head",
        "train-stld-frozen-sam-mask-head",
        "eval-stld-frozen-sam-mask-head",
        "train-caid-frozen-sam-mask-head",
        "eval-caid-frozen-sam-mask-head",
        "train-rwtd-frozen-sam-mask-head",
        "eval-rwtd-frozen-sam-mask-head",
    }:
        model_id = str(getattr(args, "model_id", "")).strip()
        if model_id != FROZEN_SAM_HEAD_ALLOWED_MODEL_ID:
            parser.error(
                f"{args.command} only supports model-id `{FROZEN_SAM_HEAD_ALLOWED_MODEL_ID}` for frozen SAM feature-head runs."
            )
    residual_gate_mode = getattr(args, "residual_gate_mode", "none")
    residual_gate_threshold = getattr(args, "residual_gate_threshold", float(RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD))
    residual_alpha_override = getattr(args, "residual_alpha_override", None)
    coarse_loss_weight = getattr(args, "coarse_loss_weight", 0.0)
    residual_l1_weight = getattr(args, "residual_l1_weight", 0.0)
    attention_sparsity_weight = getattr(args, "attention_sparsity_weight", 0.0)
    attention_hidden_dim = getattr(args, "attention_hidden_dim", int(FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"]))
    cross_attn_query_stride = getattr(args, "cross_attn_query_stride", int(FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"]))
    if residual_gate_mode not in RESIDUAL_HEAD_GATE_MODES:
        parser.error(f"--residual-gate-mode must be one of {', '.join(RESIDUAL_HEAD_GATE_MODES)}.")
    if residual_gate_mode != "none" and not is_coarse_plus_residual_variant(str(getattr(args, "variant", ""))):
        parser.error("--residual-gate-mode is only valid for coarse-plus-residual frozen mask-head variants.")
    if residual_alpha_override is not None and not is_coarse_plus_residual_variant(str(getattr(args, "variant", ""))):
        parser.error("--residual-alpha-override is only valid for coarse-plus-residual frozen mask-head variants.")
    if coarse_loss_weight < 0.0:
        parser.error("--coarse-loss-weight must be non-negative.")
    if residual_l1_weight < 0.0:
        parser.error("--residual-l1-weight must be non-negative.")
    if attention_sparsity_weight < 0.0:
        parser.error("--attention-sparsity-weight must be non-negative.")
    if attention_hidden_dim < 1:
        parser.error("--attention-hidden-dim must be at least 1.")
    if residual_gate_mode != "none" and not (0.0 < float(residual_gate_threshold) <= 0.5):
        parser.error("--residual-gate-threshold must lie in the open interval (0, 0.5].")
    if (coarse_loss_weight > 0.0 or residual_l1_weight > 0.0 or attention_sparsity_weight > 0.0) and not is_coarse_plus_residual_variant(str(getattr(args, "variant", ""))):
        parser.error("--coarse-loss-weight, --residual-l1-weight, and --attention-sparsity-weight are only valid for coarse-plus-residual frozen mask-head variants.")
    joker_disable_fpn0 = bool(getattr(args, "joker_disable_fpn0", False))
    joker_disable_fpn1 = bool(getattr(args, "joker_disable_fpn1", False))
    joker_disable_null_token = bool(getattr(args, "joker_disable_null_token", False))
    joker_use_learned_gate = bool(getattr(args, "joker_use_learned_gate", False))
    joker_zero_init_residual_scale = bool(getattr(args, "joker_zero_init_residual_scale", False))
    joker_zero_init_attn_qkv = bool(getattr(args, "joker_zero_init_attn_qkv", False))
    joker_residual_warmup_epochs = int(getattr(args, "joker_residual_warmup_epochs", 0))
    joker_residual_ramp_epochs = int(getattr(args, "joker_residual_ramp_epochs", 0))
    if any(
        (
            joker_disable_fpn0,
            joker_disable_fpn1,
            joker_disable_null_token,
            joker_use_learned_gate,
            joker_zero_init_residual_scale,
            joker_zero_init_attn_qkv,
            joker_residual_warmup_epochs > 0,
            joker_residual_ramp_epochs > 0,
        )
    ) and not is_multibank_null_refine_variant(str(getattr(args, "variant", ""))):
        #parser.error("--joker-* controls are only valid for fpn_2_plus_multibank_null_refine.")
        pass
    if is_multibank_null_refine_variant(str(getattr(args, "variant", ""))) and joker_disable_fpn0 and joker_disable_fpn1:
        parser.error("fpn_2_plus_multibank_null_refine requires at least one enabled evidence bank; do not disable both fpn_0 and fpn_1.")
    if joker_residual_warmup_epochs < 0:
        parser.error("--joker-residual-warmup-epochs must be non-negative.")
    if joker_residual_ramp_epochs < 0:
        parser.error("--joker-residual-ramp-epochs must be non-negative.")
    if joker_residual_warmup_epochs > 0 and joker_residual_ramp_epochs > 0:
        parser.error("--joker-residual-warmup-epochs and --joker-residual-ramp-epochs are mutually exclusive.")
    if attention_hidden_dim % 8 != 0:
        parser.error("--attention-hidden-dim must be divisible by 8 for the fixed GroupNorm setting in residual-attention variants.")
    if cross_attn_query_stride < 1:
        parser.error("--cross-attn-query-stride must be at least 1.")
    if getattr(args, "variant", None) in {"fpn_2_plus_fpn_1_cross_attn_refine", "fpn_2_plus_multibank_null_refine"}:
        attention_heads = int(getattr(args, "attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]))
        if attention_hidden_dim % attention_heads != 0:
            parser.error("--attention-hidden-dim must be divisible by --attention-heads for the cross-attention joker/refine heads.")

    if args.command == "inspect-dataset":
        result = inspect_dataset(
            dataset_id=args.dataset_id,
            split=args.split,
            cache_dir=args.cache_dir,
            limit=args.limit,
        )
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-one":
        result = run_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval":
        result = run_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-sam2":
        result = run_sam2_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-sam2":
        result = run_sam2_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-sam2-official":
        result = run_sam2_official_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-sam2-official":
        result = run_sam2_official_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-sam3-auto":
        result = run_sam3_auto_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-sam3-auto":
        result = run_sam3_auto_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-coarse-vs-fine-sam-probe":
        result = run_coarse_vs_fine_sam_probe_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-coarse-vs-fine-sam-probe":
        result = run_coarse_vs_fine_sam_probe_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-coarse-vs-fine-linear-probe":
        result = run_coarse_vs_fine_linear_probe_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-coarse-vs-fine-linear-probe":
        result = run_coarse_vs_fine_linear_probe_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-glas-frozen-sam-mask-head":
        result = run_glas_frozen_mask_head_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-glas-frozen-sam-mask-head":
        result = run_glas_frozen_mask_head_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-monuseg-frozen-sam-mask-head":
        result = run_monuseg_frozen_mask_head_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-monuseg-frozen-sam-mask-head":
        result = run_monuseg_frozen_mask_head_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-stld-frozen-sam-mask-head":
        result = run_stld_frozen_mask_head_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-stld-frozen-sam-mask-head":
        result = run_stld_frozen_mask_head_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-yknd-frozen-sam-mask-head":
        result = run_yknd_frozen_mask_head_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-yknd-frozen-sam-mask-head":
        result = run_yknd_frozen_mask_head_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-caid-frozen-sam-mask-head":
        result = run_caid_frozen_mask_head_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-caid-frozen-sam-mask-head":
        result = run_caid_frozen_mask_head_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-rwtd-frozen-sam-mask-head":
        result = run_rwtd_frozen_mask_head_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-rwtd-frozen-sam-mask-head":
        result = run_rwtd_frozen_mask_head_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-stld-autosam":
        result = run_stld_autosam_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-stld-autosam":
        result = run_stld_autosam_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-caid-autosam":
        result = run_stld_autosam_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-caid-autosam":
        result = run_stld_autosam_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-rwtd-autosam":
        result = run_rwtd_autosam_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-rwtd-autosam":
        result = run_rwtd_autosam_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-monuseg-autosam-faithful":
        result = run_monuseg_autosam_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-monuseg-autosam-faithful":
        result = run_monuseg_autosam_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "train-glas-autosam-faithful":
        result = run_glas_autosam_train(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-glas-autosam-faithful":
        result = run_glas_autosam_eval(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-architexture-binary":
        result = run_architexture_binary_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-architexture-binary":
        result = run_architexture_binary_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-detexture-binary":
        result = run_detexture_binary_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-detexture-binary":
        result = run_detexture_binary_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-detexture-multi":
        result = run_detexture_multi_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-detexture-multi":
        result = run_detexture_multi_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-cstd-binary":
        result = run_cstd_binary_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-cstd-binary":
        result = run_cstd_binary_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "predict-glas-binary":
        result = run_glas_binary_predict_one(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval-glas-binary":
        result = run_glas_binary_evaluation(args)
        LOGGER.info("%s", json.dumps(result, indent=2, sort_keys=True))
        return 0

    raise RuntimeError(f"Unhandled command: {args.command}")
