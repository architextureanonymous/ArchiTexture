"""Thin loaders around the upstream AutoSAM repository.

This module deliberately avoids reimplementing the AutoSAM architecture.
Instead, it imports the upstream prompt-generator and SAM modules from the
cloned ``third_party/AutoSAM`` checkout while working around the fact that the
package root currently imports optional utilities that require unavailable
``torchvision::nms`` bindings on this machine.

The public helpers here keep the integration surface small:

- ``build_upstream_autosam_prompt_generator()`` instantiates the original
  ``ModelEmb`` prompt generator.
- ``build_upstream_autosam_sam()`` instantiates the vendored SAM ViT model
  from the upstream repo and loads an official Meta checkpoint.
- ``load_upstream_autosam_transforms_shir()`` and
  ``load_upstream_autosam_resize_longest_side_class()`` expose the upstream
  paired transform utilities used by the wrapper trainers.
"""

from __future__ import annotations

from functools import lru_cache
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
AUTOSAM_REPO_ROOT = REPO_ROOT / "third_party" / "AutoSAM"
AUTOSAM_SEGMENT_ANYTHING_ROOT = AUTOSAM_REPO_ROOT / "segment_anything"
AUTOSAM_SEGMENT_ANYTHING_ALIAS = "_rwtd_autosam_segment_anything"


class AutoSamImportError(RuntimeError):
    """Raised when the upstream AutoSAM checkout cannot be loaded safely."""


def resolve_autosam_repo_root() -> Path:
    """Return the cloned upstream AutoSAM repo root or raise loudly."""

    if not AUTOSAM_REPO_ROOT.exists():
        raise AutoSamImportError(
            "Missing upstream AutoSAM checkout. Expected to find it at "
            f"'{AUTOSAM_REPO_ROOT}'. Clone https://github.com/talshaharabany/AutoSAM first."
        )
    return AUTOSAM_REPO_ROOT


def _ensure_autosam_repo_on_syspath() -> None:
    repo_root = str(resolve_autosam_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _ensure_segment_anything_alias_package() -> str:
    resolve_autosam_repo_root()
    if AUTOSAM_SEGMENT_ANYTHING_ALIAS not in sys.modules:
        module = types.ModuleType(AUTOSAM_SEGMENT_ANYTHING_ALIAS)
        module.__path__ = [str(AUTOSAM_SEGMENT_ANYTHING_ROOT)]
        sys.modules[AUTOSAM_SEGMENT_ANYTHING_ALIAS] = module
    utils_alias = f"{AUTOSAM_SEGMENT_ANYTHING_ALIAS}.utils"
    if utils_alias not in sys.modules:
        module = types.ModuleType(utils_alias)
        module.__path__ = [str(AUTOSAM_SEGMENT_ANYTHING_ROOT / "utils")]
        sys.modules[utils_alias] = module
    return AUTOSAM_SEGMENT_ANYTHING_ALIAS


def _load_segment_anything_submodule(relative_module_name: str):
    package_alias = _ensure_segment_anything_alias_package()
    module_name = f"{package_alias}.{relative_module_name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    relative_path = Path(*relative_module_name.split(".")).with_suffix(".py")
    module_path = AUTOSAM_SEGMENT_ANYTHING_ROOT / relative_path
    if not module_path.exists():
        raise AutoSamImportError(
            f"Upstream AutoSAM submodule '{relative_module_name}' does not exist at '{module_path}'."
        )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AutoSamImportError(f"Failed to build an import spec for '{module_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_upstream_autosam_model_emb_class():
    """Return the upstream ``ModelEmb`` class."""

    _ensure_autosam_repo_on_syspath()
    module = importlib.import_module("models.model_single")
    return getattr(module, "ModelEmb")


@lru_cache(maxsize=1)
def load_upstream_autosam_transforms_shir():
    """Return the upstream paired image/mask transform module."""

    _ensure_autosam_repo_on_syspath()
    return importlib.import_module("dataset.transforms_shir")


@lru_cache(maxsize=1)
def load_upstream_autosam_resize_longest_side_class():
    """Return the vendored SAM ``ResizeLongestSide`` class."""
    try:
        module = _load_segment_anything_submodule("utils.transforms")
        return getattr(module, "ResizeLongestSide")
    except Exception:
        class _ResizeLongestSide:
            def __init__(self, target_length: int) -> None:
                self.target_length = int(target_length)

            @staticmethod
            def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
                scale = float(long_side_length) / float(max(oldh, oldw))
                return max(1, int(round(oldh * scale))), max(1, int(round(oldw * scale)))

            def apply_image(self, image: Any):
                from PIL import Image

                if image.ndim != 3:
                    raise AutoSamImportError("Fallback ResizeLongestSide expects an HxWxC image array.")
                h, w = int(image.shape[0]), int(image.shape[1])
                newh, neww = self.get_preprocess_shape(h, w, self.target_length)
                pil = Image.fromarray(image.astype("uint8"))
                resized = pil.resize((neww, newh), resample=Image.Resampling.BILINEAR)
                return np.asarray(resized, dtype=image.dtype)

        return _ResizeLongestSide


@lru_cache(maxsize=1)
def load_upstream_autosam_sam_registry() -> dict[str, Any]:
    """Return the vendored SAM registry without importing the broken package root."""

    module = _load_segment_anything_submodule("build_sam")
    registry = getattr(module, "sam_model_registry")
    if not isinstance(registry, dict):
        raise AutoSamImportError("Upstream AutoSAM build_sam.py did not expose a sam_model_registry dict.")
    return registry


def build_upstream_autosam_prompt_generator(
    *,
    order: int,
    depth_wise: bool,
) -> Any:
    """Instantiate the original AutoSAM prompt-generator model."""

    model_class = load_upstream_autosam_model_emb_class()
    args = {
        "order": int(order),
        "depth_wise": bool(depth_wise),
    }
    return model_class(args=args)


def build_upstream_autosam_sam(
    *,
    model_type: str,
    checkpoint_path: str | Path,
) -> Any:
    """Instantiate the vendored SAM model and load an official checkpoint."""

    registry = load_upstream_autosam_sam_registry()
    resolved_model_type = str(model_type)
    if resolved_model_type not in registry:
        raise AutoSamImportError(
            f"Unsupported SAM model_type '{model_type}'. Available registry keys: {sorted(registry)}"
        )
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint path does not exist: {resolved_checkpoint_path}")
    builder = registry[resolved_model_type]
    return builder(checkpoint=str(resolved_checkpoint_path))
