"""Train-time augmentations for frozen-feature dense mask-head baselines."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.eval.metrics import boundary_from_region_masks
from rwtd_sam3.models.sam3_frozen_multiscale_mask_head import FrozenMaskHeadRuntimeError


FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES = ("none", "autosam_dense_v1", "monuseg_paper_v1")
AUTOSAM_DENSE_AUGMENTATION_SETTINGS: dict[str, float] = {
    "rotation_degrees": 20.0,
    "scale_min": 0.75,
    "scale_max": 1.25,
    "horizontal_flip_probability": 0.5,
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.4,
    "hue": 0.1,
}


def resolve_train_augmentation_policy(args) -> str:
    return str(getattr(args, "train_augmentation_policy", "none"))


def describe_train_augmentation_policy(policy: str) -> str:
    if policy == "none":
        return "none"
    if policy not in ("autosam_dense_v1", "monuseg_paper_v1"):
        raise FrozenMaskHeadRuntimeError(
            f"Unsupported train augmentation policy '{policy}'.",
            diagnostics={"supported_policies": list(FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES)},
        )
    policy_prefix = "monuseg_paper_v1" if policy == "monuseg_paper_v1" else "autosam_dense_v1"
    return (
        f"{policy_prefix}:"
        "affine(angle_uniform[-20,20],scale_uniform[0.75,1.25],translate=0,shear=0,fill=0)"
        "+hflip_p0.5"
        "+color_jitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)"
        "+train_only"
    )


def apply_train_augmentation_to_binary_sample(
    sample: Any,
    *,
    rng: np.random.Generator,
    policy: str,
    max_attempts: int = 4,
) -> Any:
    """Apply one train-time augmentation policy to a binary segmentation sample."""

    if policy == "none":
        return sample
    if policy not in ("autosam_dense_v1", "monuseg_paper_v1"):
        raise FrozenMaskHeadRuntimeError(
            f"Unsupported train augmentation policy '{policy}'.",
            diagnostics={"supported_policies": list(FROZEN_MASK_HEAD_TRAIN_AUGMENTATION_POLICIES)},
        )
    for _ in range(max(1, int(max_attempts))):
        augmented = _apply_autosam_dense_v1(sample=sample, rng=rng)
        foreground = np.asarray(augmented.texture_a_mask, dtype=bool)
        background = np.asarray(augmented.texture_b_mask, dtype=bool)
        if foreground.any() and background.any():
            return augmented
    return sample


def _apply_autosam_dense_v1(sample: Any, *, rng: np.random.Generator) -> Any:
    angle = float(rng.uniform(-AUTOSAM_DENSE_AUGMENTATION_SETTINGS["rotation_degrees"], AUTOSAM_DENSE_AUGMENTATION_SETTINGS["rotation_degrees"]))
    scale = float(
        rng.uniform(
            AUTOSAM_DENSE_AUGMENTATION_SETTINGS["scale_min"],
            AUTOSAM_DENSE_AUGMENTATION_SETTINGS["scale_max"],
        )
    )
    do_hflip = bool(rng.random() < AUTOSAM_DENSE_AUGMENTATION_SETTINGS["horizontal_flip_probability"])

    foreground_mask = np.asarray(sample.texture_a_mask, dtype=np.uint8) * 255
    foreground_mask_image = Image.fromarray(foreground_mask, mode="L")

    image = _apply_affine_pil(sample.image, angle=angle, scale=scale, resample=Image.Resampling.BILINEAR)
    mask_image = _apply_affine_pil(foreground_mask_image, angle=angle, scale=scale, resample=Image.Resampling.NEAREST)
    if do_hflip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    brightness_factor = float(
        rng.uniform(
            max(0.0, 1.0 - AUTOSAM_DENSE_AUGMENTATION_SETTINGS["brightness"]),
            1.0 + AUTOSAM_DENSE_AUGMENTATION_SETTINGS["brightness"],
        )
    )
    contrast_factor = float(
        rng.uniform(
            max(0.0, 1.0 - AUTOSAM_DENSE_AUGMENTATION_SETTINGS["contrast"]),
            1.0 + AUTOSAM_DENSE_AUGMENTATION_SETTINGS["contrast"],
        )
    )
    saturation_factor = float(
        rng.uniform(
            max(0.0, 1.0 - AUTOSAM_DENSE_AUGMENTATION_SETTINGS["saturation"]),
            1.0 + AUTOSAM_DENSE_AUGMENTATION_SETTINGS["saturation"],
        )
    )
    hue_factor = float(
        rng.uniform(
            -AUTOSAM_DENSE_AUGMENTATION_SETTINGS["hue"],
            AUTOSAM_DENSE_AUGMENTATION_SETTINGS["hue"],
        )
    )
    jitter_ops: list[tuple[str, float]] = [
        ("brightness", brightness_factor),
        ("contrast", contrast_factor),
        ("saturation", saturation_factor),
        ("hue", hue_factor),
    ]
    for op_index in rng.permutation(len(jitter_ops)).tolist():
        op_name, op_value = jitter_ops[int(op_index)]
        if op_name == "brightness":
            image = _adjust_brightness_pil(image, op_value)
        elif op_name == "contrast":
            image = _adjust_contrast_pil(image, op_value)
        elif op_name == "saturation":
            image = _adjust_saturation_pil(image, op_value)
        elif op_name == "hue":
            image = _adjust_hue_pil(image, op_value)
        else:
            raise AssertionError(f"Unexpected color-jitter op '{op_name}'.")

    augmented_foreground_mask = np.asarray(mask_image, dtype=np.uint8) > 0
    augmented_background_mask = np.logical_not(augmented_foreground_mask)
    augmented_boundary_mask = boundary_from_region_masks(augmented_foreground_mask, augmented_background_mask)
    return replace(
        sample,
        image=image,
        boundary_mask=np.asarray(augmented_boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(augmented_foreground_mask, dtype=bool),
        texture_b_mask=np.asarray(augmented_background_mask, dtype=bool),
    )


def _apply_affine_pil(image: Image.Image, *, angle: float, scale: float, resample: int) -> Image.Image:
    """Apply the narrow affine transform subset used by the frozen-mask-head augmentations."""

    width, height = image.size
    center_x = width / 2.0
    center_y = height / 2.0
    rad = np.deg2rad(angle)
    cos_t = float(np.cos(rad) * scale)
    sin_t = float(np.sin(rad) * scale)
    a = cos_t
    b = sin_t
    c = (1.0 - a) * center_x - b * center_y
    d = -sin_t
    e = cos_t
    f = b * center_x + (1.0 - e) * center_y
    return image.transform((width, height), Image.Transform.AFFINE, (a, b, c, d, e, f), resample=resample, fillcolor=0)


def _adjust_brightness_pil(image: Image.Image, factor: float) -> Image.Image:
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(image).enhance(float(factor))


def _adjust_contrast_pil(image: Image.Image, factor: float) -> Image.Image:
    from PIL import ImageEnhance

    return ImageEnhance.Contrast(image).enhance(float(factor))


def _adjust_saturation_pil(image: Image.Image, factor: float) -> Image.Image:
    from PIL import ImageEnhance

    return ImageEnhance.Color(image).enhance(float(factor))


def _adjust_hue_pil(image: Image.Image, factor: float) -> Image.Image:
    import colorsys

    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in flat], dtype=np.float32)
    hsv[:, 0] = np.mod(hsv[:, 0] + float(factor), 1.0)
    rgb_out = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in hsv], dtype=np.float32)
    rgb_out = np.clip(rgb_out.reshape(rgb.shape) * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb_out, mode="RGB")
