# Evaluation Notes

This directory contains the evaluation entrypoints for every experiment family in the repository:

- prompt-conditioned RWTD SAM 3 in [`runner.py`](./runner.py)
- crop-format RWTD SAM-2 baselines in [`sam2_baseline.py`](./sam2_baseline.py)
- paper-faithful official TextureSAM RWTD reproduction in [`sam2_official_rwtd.py`](./sam2_official_rwtd.py)
- SAM-3 automatic-mask baselines and feature ablations in [`sam3_auto.py`](./sam3_auto.py)
- Stage-2 coarse-vs-fine SAM scale study train/eval entrypoints in [`coarse_vs_fine_sam_scales.py`](./coarse_vs_fine_sam_scales.py)
- local ArchiTexture STLD / CAID route support in [`architexture_binary.py`](./architexture_binary.py)
- local DeTexture ADE20K binary support in [`detexture_binary.py`](./detexture_binary.py)
- local DeTexture ADE20K multi-region support in [`detexture_multi.py`](./detexture_multi.py)
- local CSTD binary support in [`cstd_binary.py`](./cstd_binary.py)
- local GlaS binary support in [`glas_binary.py`](./glas_binary.py)
- dense-supervised frozen-SAM GlaS baselines in [`glas_frozen_feature_mask_head.py`](./glas_frozen_feature_mask_head.py)
- cross-dataset current-method experiment registry in [`experiment_registry.py`](./experiment_registry.py)

The root [`README.md`](./README.md) is the operator guide. This file is the short semantic map for the evaluation modules themselves.

The repo-wide evaluation policy now lives in
[`EVALUATION_CONTRACT.md`](./EVALUATION_CONTRACT.md).
That document is authoritative for split-role fairness, protocol families,
route-primary metrics, required output artifacts, and the planned central
evaluation entry point migration.

## experiment_terms.md Standard

Every evaluation-producing experiment is expected to write an experiment-specific `experiment_terms.md` into its results directory. That file is not a generic glossary. It should explain the exact experiment that produced that run, including:

- hypothesis
- scientific idea / pipeline
- dataset and split separation (`train` / `eval` / `test`, or explicit note that no validation split exists)
- architecture or probe details when learning is involved
- feature source, alignment policy, and tensor/level choices when relevant
- training objective and optimizer settings when relevant
- evaluation path and metric semantics
- output artifact structure
- hard-failure conditions and no-fallback behavior

The shared renderer for this richer per-run contract lives in [`experiment_terms.py`](./experiment_terms.py). New experiment families should use that standard instead of writing a thin runtime appendix.
The research workspace for method iteration lives under
[`research/`](./),
with the current active runbook in
[`research/current_method/README.md`](./README.md).

## Legacy Shared Comparison View

Most evaluation-producing paths in this directory still export the same shared
binary comparison view:

- `evaluation_contract = architexture_binary_v1`
- primary metrics: `eval_miou` and `eval_ari`

Those paths map their own predictions into a canonical binary partition before those defaults are computed. Path-specific metrics remain in the artifacts as diagnostics.

This legacy shared view is still useful for backward-compatible comparison, but
it is no longer the full repo-wide evaluation contract. The repo-wide contract
now also requires:

- validation-only checkpoint selection
- test-only headline reporting
- explicit protocol-family declaration
- explicit route-primary metric declaration
- machine-recorded deviations and safety tags

The exception is the multi-region DeTexture route in [`detexture_multi.py`](./detexture_multi.py):

- `evaluation_contract = detexture_multi_partition_v1`
- primary metrics: Hungarian-matched `eval_miou`, `eval_ari`, and `eval_nmi` on valid pixels
- region-count agreement is reported as `count_accuracy`
- supported multi-region partition heads currently include deterministic oracle-K / predicted-K clustering, the trainable `feature_cluster_coarse_global_pooled_deepdpm` and `..._flipavg` ablations, and the simple `feature_cluster_coarse_global_pooled_hdbscan` and `..._flipavg` ablations on the same frozen pooled SAM features

## Hardware Compatibility Golden Standard

Hardware-aware throughput is a repository requirement for evaluation code paths.

- New evaluation entrypoints must not default to sample-by-sample GPU starvation when a batched inference path exists.
- New evaluation entrypoints must expose throughput controls at the CLI layer and persist resolved settings into run artifacts.
- CUDA-oriented eval defaults must use a worker-backed dataloader plus explicit host-to-device staging behavior.

For prompt-conditioned RWTD SAM 3 in [`runner.py`](./runner.py), this standard is tracked as `eval_throughput_v1`:

- `batch_size=8`
- `num_workers=min(8, os.cpu_count())`
- `pin_memory=true`
- `prefetch_factor=2`

Those settings are saved into `config.json` and restated in `experiment_terms.md`.

For the ArchiTexture route adapter in [`architexture_binary.py`](./architexture_binary.py), the route-facing standard is `architexture_streaming_eval_v1`:

- samples are streamed instead of preloaded into a Python list
- the benchmark root is resolved once per run, not once per sample
- the active `sample_loading_mode` is persisted into `config.json`

For the DeTexture ADE20K adapter in [`detexture_binary.py`](./detexture_binary.py), the dataset-facing standard is `detexture_streaming_eval_v1`:

- samples are streamed instead of preloaded into a Python list
- the dataset root is resolved once per run, not once per sample
- the active `sample_loading_mode` is persisted into `config.json`

For the DeTexture ADE20K multi-region adapter in [`detexture_multi.py`](./detexture_multi.py), the dataset-facing standard is `detexture_multi_streaming_eval_v1`:

- samples are streamed instead of preloaded into a Python list
- the `detecture_data` root is resolved once per run, not once per sample
- the active `sample_loading_mode` is persisted into `config.json`
- GT decode settings and the multi-region evaluation contract are persisted into run artifacts

For the CSTD adapter in [`cstd_binary.py`](./cstd_binary.py), the dataset-facing standard is `cstd_streaming_eval_v1`:

- samples are streamed instead of preloaded into a Python list
- the dataset root is resolved once per run, not once per sample
- the active `sample_loading_mode` is persisted into `config.json`

For the GlaS adapter in [`glas_binary.py`](./glas_binary.py), the dataset-facing standard is `glas_streaming_eval_v1`:

- samples are streamed instead of preloaded into a Python list
- the dataset root is resolved once per run, not once per sample
- the active `sample_loading_mode` is persisted into `config.json`

For the dense-supervised frozen-SAM GlaS path in [`glas_frozen_feature_mask_head.py`](./glas_frozen_feature_mask_head.py):

- primary supervised metrics are `direct_foreground_iou` and `direct_foreground_dice`
- auxiliary repo-native metrics remain `eval_miou` and `eval_ari` under `architexture_binary_v1`
- the backbone stays frozen; only the tiny head is optimized
- large full-split runs can switch from in-memory feature caching to `materialization_mode=streamed_reextract` when the estimated raw feature cache exceeds the internal `5 GiB` budget

## SAM-3 Automatic-Mask Variant Index

| Variant | What it is doing | Requires official Meta `sam3` |
| --- | --- | --- |
| `default` | Plain Transformers `mask-generation` baseline with the lighter proposal density. | No |
| `dense` | Plain Transformers `mask-generation` baseline with denser proposals and a lower stability threshold. | No |
| `feature_mask` | Finds one adjacent dense-mask pair, pools official SAM dense features inside it, builds a local coarse prior, then refines through SAM mask prompting. | Yes |
| `feature_cluster_global` | Clusters normalized official SAM dense features globally into two unlabeled groups and refines both clusters with SAM. | Yes |
| `feature_cluster_coarse_to_fine_global` | Initializes on the coarsest feature level, refines only near the current boundary at finer levels, then refines with SAM. | Yes |
| `feature_cluster_coarse_to_fine_global_pooled_init` | Same as the coarse-to-fine variant, but average-pools the coarsest feature map before the initial 2-cluster split. | Yes |
| `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only` | Uses the pooled coarsest partition directly as the final prediction with no finer-level refinement and no SAM prompt refinement. | Yes |
| `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` | Averages coarsest features across `identity/hflip/vflip/hvflip`, then runs the pooled coarse-only partition directly as the final prediction. | Yes |
| `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only` | Position-leakage diagnosis branch: measures raw coarsest positionality, removes the coordinate projection before pooled clustering, compares baseline/flip-averaged branches, and keeps the projection-removed coarse partition as the active output. | Yes |
| `boundary_refine_sweep` | Keeps the strong flip-averaged pooled coarse partition fixed and compares five conservative boundary-only refinement variants V0-V4 on top of it. | Yes |
| `flip_avg_plus_edge_debias` | Residual edge-bias side experiment: measures flip-symmetric border leakage, compares baseline/flip-only/edge-only/flip+edge pooled branches, and keeps the combined branch as the active output. | Yes |
| `feature_cluster_coarse_to_fine_global_pooled_init_direct` | Skips finer-level feature refinement, but still feeds the pooled coarsest partition into SAM as the mask prompt. | Yes |
| `mask_prompt_invariance_control` | Diagnostic side experiment comparing dirty/raw, cleaner/pooled, and random prompt branches under the exact same SAM refinement path. | Yes |

Additional route note:

- `predict-architexture-binary` and `eval-architexture-binary` now accept any experiment registered in [`experiment_registry.py`](./experiment_registry.py).
- The default remains `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only`.

## Coarse-vs-Fine SAM Scale Study Stage 2

This Stage-2 path keeps SAM frozen, aligns the main `backbone_fpn` levels to the coarsest selected grid, learns only a tiny per-level `1x1` projection plus global scale gates, and still evaluates by `k=2` deterministic clustering of the learned embedding.

Runnable Stage-2 variants:

| Variant | Commands | What it does |
| --- | --- | --- |
| `learned_global_gates_all_scales` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Learns global softmax scale weights over all discovered main SAM pyramid levels. |
| `fixed_uniform_gates_all_scales` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Uses the same tiny probe but keeps the scale weights fixed uniform. |
| `learned_global_gates_coarse_plus_next_finer` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Learns global softmax scale weights using only the coarsest level and the next finer level. |
| `fixed_uniform_gates_coarse_plus_next_finer` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Uses the same two-level coarse-plus-next-finer probe but keeps the scale weights fixed uniform. |
| `fpn_2_only` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Runs the Stage-2 probe on only the coarsest discovered SAM level `fpn_2`. |
| `fpn_1_only` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Runs the Stage-2 probe on only the middle discovered SAM level `fpn_1`. |
| `fpn_0_only` | `train-coarse-vs-fine-sam-probe`, `eval-coarse-vs-fine-sam-probe` | Runs the Stage-2 probe on only the finest discovered SAM level `fpn_0`. |

Runnable few-shot linear-probe variants:

| Variant | Commands | What it does |
| --- | --- | --- |
| `concat_all_scales` | `train-coarse-vs-fine-linear-probe`, `eval-coarse-vs-fine-linear-probe` | Concatenates all discovered aligned SAM pyramid levels and fits one supervised `1x1` linear classifier. |
| `concat_coarse_plus_next_finer` | `train-coarse-vs-fine-linear-probe`, `eval-coarse-vs-fine-linear-probe` | Concatenates the coarsest level and the next finer level, then fits one supervised `1x1` linear classifier. |
| `fpn_2_only` | `train-coarse-vs-fine-linear-probe`, `eval-coarse-vs-fine-linear-probe` | Runs the few-shot linear probe on only the coarsest discovered SAM level `fpn_2`. |
| `fpn_1_only` | `train-coarse-vs-fine-linear-probe`, `eval-coarse-vs-fine-linear-probe` | Runs the few-shot linear probe on only the middle discovered SAM level `fpn_1`. |
| `fpn_0_only` | `train-coarse-vs-fine-linear-probe`, `eval-coarse-vs-fine-linear-probe` | Runs the few-shot linear probe on only the finest discovered SAM level `fpn_0`. |

Quick starters:

Default Stage-2 settings are used unless overridden: `--train-split train`, `--eval-split test`, `--num-epochs 5`, `--pairs-per-image 512`, cosine `k=2`, projection dim `64`, and embedding dim `32`.

Few-shot training is now explicit: add `--num-train-samples N` to the train command to optimize on only the first `N` samples from the requested training split.

Few-shot linear-probe quick starts:

```bash
python main.py train-coarse-vs-fine-linear-probe \
  --dataset-source rwtd \
  --variant concat_all_scales \
  --train-split train \
  --eval-split test \
  --num-train-samples 16 \
  --eval-limit 128 \
  --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/linear_probe/train_concat_all_scales_rwtd_16shot
```

```bash
python main.py train-coarse-vs-fine-linear-probe \
  --dataset-source architexture_binary \
  --route stld \
  --benchmark-root experiments/dataset_splits/stld_train32_test168_seed0/prepared_root \
  --variant concat_all_scales \
  --train-split train \
  --eval-split test \
  --num-train-samples 16 \
  --eval-limit 128 \
  --output-dir outputs/architexture_binary/coarse_vs_fine_sam_scales/linear_probe/train_concat_all_scales_stld_16shot
```

```bash
python main.py train-coarse-vs-fine-linear-probe \
  --dataset-source architexture_binary \
  --route caid \
  --benchmark-root experiments/dataset_splits/random_50_50_seed0/prepared_roots/caid \
  --variant concat_all_scales \
  --train-split train \
  --eval-split test \
  --num-train-samples 16 \
  --eval-limit 128 \
  --output-dir outputs/architexture_binary/coarse_vs_fine_sam_scales/linear_probe/train_concat_all_scales_caid_16shot
```

```bash
python main.py train-coarse-vs-fine-linear-probe \
  --dataset-source cstd_binary \
  --dataset-root experiments/dataset_splits/random_50_50_seed0/prepared_roots/cstd \
  --variant concat_all_scales \
  --train-split train \
  --eval-split test \
  --num-train-samples 16 \
  --eval-limit 128 \
  --output-dir outputs/cstd_binary/coarse_vs_fine_sam_scales/linear_probe/train_concat_all_scales_cstd_16shot
```

STLD official-split quick start:

- Stage-2 STLD runs require an official split-aware local root. Do not point these commands at the flat `datasets/STLD` benchmark export.
- Accepted layouts are either Pascal-VOC-style `ImageSets/Segmentation/{train,test}.txt` with `JPEGImages/` and `SegmentationClass/`, or split subdirectories like `train/images`, `train/labels`, `test/images`, and `test/labels`.

```bash
python main.py train-coarse-vs-fine-sam-probe   --dataset-source architexture_binary   --route stld   --benchmark-root /path/to/STLD   --variant learned_global_gates_all_scales   --train-split train   --eval-split test   --output-dir outputs/architexture_binary/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_stld
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --dataset-source architexture_binary   --route stld   --benchmark-root /path/to/STLD   --checkpoint-path outputs/architexture_binary/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_stld/checkpoint.pt   --split test   --output-dir outputs/architexture_binary/coarse_vs_fine_sam_scales/stage2/eval_learned_global_gates_all_scales_stld
```

CAID official-split quick start:

- The local CAID official split root already present in this workspace is
  `datasets/architexture/CAID`.

```bash
python main.py train-coarse-vs-fine-sam-probe \
  --dataset-source architexture_binary \
  --route caid \
  --benchmark-root datasets/architexture/CAID \
  --variant learned_global_gates_all_scales \
  --train-split train \
  --eval-split test \
  --output-dir outputs/architexture_binary/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_caid
```

```bash
python main.py eval-coarse-vs-fine-sam-probe \
  --dataset-source architexture_binary \
  --route caid \
  --benchmark-root datasets/architexture/CAID \
  --checkpoint-path outputs/architexture_binary/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_caid/checkpoint.pt \
  --split test \
  --output-dir outputs/architexture_binary/coarse_vs_fine_sam_scales/stage2/eval_learned_global_gates_all_scales_caid
```

CSTD official-split quick start:

- Stage-2 on CSTD must use an official split-aware local root, not the flat
  `datasets/CSTD` benchmark export currently present in this workspace.
- Supported layouts are either:
  - `ImageSets/Segmentation/{train,test,val}.txt` plus flat `images/`,
    `regions/`, and `edges/`
  - split subdirectories like `train/images`, `train/regions`, `train/edges`,
    `test/images`, `test/regions`, and `test/edges`

```bash
python main.py train-coarse-vs-fine-sam-probe \
  --dataset-source cstd_binary \
  --dataset-root /path/to/CSTD \
  --variant learned_global_gates_all_scales \
  --train-split train \
  --eval-split test \
  --output-dir outputs/cstd_binary/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_cstd
```

```bash
python main.py eval-coarse-vs-fine-sam-probe \
  --dataset-source cstd_binary \
  --dataset-root /path/to/CSTD \
  --checkpoint-path outputs/cstd_binary/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_cstd/checkpoint.pt \
  --split test \
  --output-dir outputs/cstd_binary/coarse_vs_fine_sam_scales/stage2/eval_learned_global_gates_all_scales_cstd
```

Few-shot helper scripts for Stage-2:

- Use the dataset-specific wrappers under [`scripts/`](./) when you want a fixed few-shot Stage-2 entrypoint with custom train size.
- They wrap `train-coarse-vs-fine-sam-probe` only, because the train entrypoint already performs the post-training evaluation on `--eval-split`.
- CAID defaults to the real local root [`datasets/architexture/CAID`](./).
- STLD and CSTD still require explicit official split-aware roots.

```bash
bash scripts/run_stage2_few_shot_rwtd.sh \
  --num-train-samples 16
```

```bash
bash scripts/run_stage2_few_shot_caid.sh \
  --num-train-samples 16
```

```bash
STLD_ROOT=/path/to/STLD \
bash scripts/run_stage2_few_shot_stld.sh \
  --num-train-samples 16
```

```bash
CSTD_ROOT=/path/to/CSTD \
bash scripts/run_stage2_few_shot_cstd.sh \
  --num-train-samples 16
```

```bash
bash scripts/run_stage2_few_shot_rwtd.sh \
  --num-train-samples 16 \
  --variant fixed_uniform_gates_coarse_plus_next_finer \
  --dry-run
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant learned_global_gates_all_scales   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_all_scales_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_learned_global_gates_all_scales_test
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant fixed_uniform_gates_all_scales   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fixed_uniform_gates_all_scales_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fixed_uniform_gates_all_scales_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_fixed_uniform_gates_all_scales_test
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant learned_global_gates_coarse_plus_next_finer   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_coarse_plus_next_finer_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_learned_global_gates_coarse_plus_next_finer_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_learned_global_gates_coarse_plus_next_finer_test
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant fixed_uniform_gates_coarse_plus_next_finer   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fixed_uniform_gates_coarse_plus_next_finer_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fixed_uniform_gates_coarse_plus_next_finer_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_fixed_uniform_gates_coarse_plus_next_finer_test
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant fpn_2_only   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_2_only_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_2_only_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_fpn_2_only_test
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant fpn_1_only   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_1_only_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_1_only_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_fpn_1_only_test
```

```bash
python main.py train-coarse-vs-fine-sam-probe   --variant fpn_0_only   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_0_only_test
```

```bash
python main.py eval-coarse-vs-fine-sam-probe   --checkpoint-path outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/train_fpn_0_only_test/checkpoint.pt   --output-dir outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/eval_fpn_0_only_test
```

Output root:

- Stage-2 train/eval runs write under `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/`.
- Train runs add `checkpoint.pt`, `gate_weights.json`, and `train_history.csv` on top of the standard eval artifacts.
- Eval runs require `--checkpoint-path` and re-use the exact Stage-1 invariant `eval_miou` / `eval_ari` semantics after k-means on the learned embedding.

## Cross-Dataset Experiment Contract

- Current-method experiments that are intended to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS must be registered once in [`experiment_registry.py`](./experiment_registry.py).
- A registered experiment must declare its supported datasets, runner class, and visual contract.
- Once registered, it must be documented in the root [`README.md`](./README.md) and become selectable from `eval-sam3-auto`, `eval-architexture-binary`, `eval-detexture-binary`, `eval-cstd-binary`, and `eval-glas-binary` where the registry applies.
- Experiments that are not registered there are treated as RWTD-only until they are explicitly promoted.
- Every eval-facing current-method surface also supports `--dataset-partition K/N`, applied deterministically by dataset order before `--limit`.
- The DeTexture multi-region ablation is intentionally separate from this binary registry contract; it reuses the pooled-coarsest method family but has its own variable-K dataset/eval modules.

## Comparison Views

| Evaluation path | Primary metrics | Why |
| --- | --- | --- |
| Prompt-conditioned RWTD SAM 3 | `eval_miou`, `eval_ari`, boundary metrics | The direct A/B predictions already match the canonical binary-partition view. |
| RWTD crop-format SAM-2 | `eval_miou`, `eval_ari` | Raw mask sets are collapsed into one canonical aggregated A/B partition before scoring. |
| Official TextureSAM RWTD | `eval_miou`, `eval_ari` by default; released TextureSAM metrics retained | The default repo view is canonical, while the released-script metrics remain for auditability. |
| ArchiTexture STLD | `eval_miou`, `eval_ari` | The route is treated as a foreground/background task. |
| ArchiTexture CAID | `eval_miou`, `eval_ari` | The route scores both cluster-to-region assignments and keeps the better one. |
| DeTexture ADE20K | `eval_miou`, `eval_ari` | The crops expose paired `mask_a` / `mask_b` supervision, so both cluster-to-mask assignments are scored and the better one is kept. |
| DeTexture ADE20K multi-region | Hungarian-matched `eval_miou`, `eval_ari`, `eval_nmi` | Prompt-free pooled coarsest K-way clustering is scored against a decoded variable-K GT label map with label-permutation-invariant matching; GT regions below 12% of the crop are treated as joker/void and excluded. |
| CSTD | `eval_miou`, `eval_ari` | Each sample exposes one binary `regions/` mask plus its complement, so both cluster-to-region assignments are scored and the better one is kept. |
| GlaS | `eval_miou`, `eval_ari` | Each sample exposes one binary gland mask plus its complement, so both cluster-to-region assignments are scored and the better one is kept. |

## Output Behavior

- Prompt-conditioned SAM 3 writes under `outputs/rwtd_sam3/`.
- Crop-format SAM-2 writes under `outputs/rwtd_sam2/`.
- Official TextureSAM RWTD writes under `outputs/rwtd_sam2_official/`.
- SAM-3 automatic-mask runs write under `outputs/rwtd_sam3_auto/`.
- Coarse-vs-fine Stage-2 probe runs write under `outputs/rwtd_sam3_auto/coarse_vs_fine_sam_scales/stage2/`.
- ArchiTexture route runs write under `outputs/architexture_binary/`.
- DeTexture ADE20K runs write under `outputs/detexture_binary/`.
- DeTexture ADE20K multi-region runs write under `outputs/detexture_multi/`.
- CSTD runs write under `outputs/cstd_binary/`.
- GlaS runs write under `outputs/glas_binary/`.

Shared artifacts:

- `config.json`
- `experiment_terms.md`
- `prediction.json` and `prediction.png` for predict commands
- `per_sample_metrics.csv`, `summary.json`, `summary.md`, and optional `visuals/*.png` for eval commands
- `visuals_manifest.jsonl` when previews are saved

Prompt-conditioned RWTD SAM 3 also persists the active hardware-throughput standard and dataloader settings in `config.json`, `summary.json`, and `experiment_terms.md`.

Validation behavior:

- `eval-sam3-auto` and `eval-architexture-binary` refuse to mix predict and eval artifacts in the same output directory.
- Those same command families also refuse to reuse a directory that already belongs to a different variant or route.
- The prompt-conditioned SAM 3 and SAM-2 paths write directly into the requested directory, so use fresh run directories when you want strict separation.

## Failure Semantics

- Missing optional runtime dependencies are surfaced as explicit runtime errors.
- Invalid split names, invalid `--limit` values, and invalid `--log-every` values are rejected at the CLI layer before inference starts.
- Missing RWTD cache data, missing local benchmark roots, unmatched image/label files, invalid mask shapes, and malformed oracle-point fields all fail loudly.
- Feature-based SAM-3 variants do not silently fall back to `dense` or to another refinement path. Failures are recorded directly in the run summaries when `failure_policy=skip` is active.

## Where To Look In Code

- Run configuration and artifact writers: [`runner.py`](./runner.py)
- Shared metric definitions: [`metrics.py`](./metrics.py)
- Automatic-mask output directory validation: [`sam3_auto.py`](./sam3_auto.py)
- ArchiTexture route-specific scoring and output validation: [`architexture_binary.py`](./architexture_binary.py)
- DeTexture ADE20K scoring and output validation: [`detexture_binary.py`](./detexture_binary.py)
- DeTexture ADE20K multi-region scoring and output validation: [`detexture_multi.py`](./detexture_multi.py)
- CSTD scoring and output validation: [`cstd_binary.py`](./cstd_binary.py)
- GlaS scoring and output validation: [`glas_binary.py`](./glas_binary.py)

## SAM-2 Vanilla CFC Ablation

Registered cross-dataset variant:
- `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2`

Contract:
- keep the vanilla binary CFC head unchanged
- swap only the frozen backbone from SAM-3 to SAM-2
- use the SAM-2 coarsest image embedding as the frozen feature map
- keep pooled coarse-only partitioning, nearest-neighbor upsampling, and binary evaluation unchanged

Recommended command:
```bash
python main.py eval-detexture-binary \
  --dataset-root datasets/detexture_ADE20K \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --model-id facebook/sam2-hiera-small \
  --device cuda \
  --output-dir outputs/research/current_method/detexture_binary/sam2_vanilla_cfc
```
