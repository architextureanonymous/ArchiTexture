import unittest

import cv2
import numpy as np

from texturesam_v2.consolidator import ConsolidationConfig, TextureSAMV2Consolidator
from texturesam_v2.merge import MergeConfig


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return 1.0 if uni == 0 else float(inter / uni)


class TestConsolidator(unittest.TestCase):
    def _synthetic_scene(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        h, w = 128, 128
        yy, xx = np.mgrid[0:h, 0:w]

        # Background texture.
        bg = ((np.sin(xx / 6.0) + np.cos(yy / 8.0)) * 20 + 120).astype(np.float32)
        image = np.stack([bg, bg + 6, bg - 6], axis=2)

        # Foreground textured disk.
        cx, cy, r = 64, 64, 30
        gt = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)
        fg = ((np.sin((xx + yy) / 3.0) * 35) + 170).astype(np.float32)
        for c in range(3):
            image[..., c] = np.where(gt > 0, fg + (c - 1) * 5.0, image[..., c])

        image = np.clip(image, 0, 255).astype(np.uint8)

        # Fragmented foreground proposals.
        left = np.logical_and(gt > 0, xx < cx + 2).astype(np.uint8)
        right = np.logical_and(gt > 0, xx > cx - 2).astype(np.uint8)
        upper = np.logical_and(gt > 0, yy < cy + 2).astype(np.uint8)

        # Distractor with different texture and weak overlap with gt.
        distractor = np.zeros((h, w), dtype=np.uint8)
        distractor[20:46, 80:112] = 1

        proposals = [left, right, upper, distractor]
        return image, gt, proposals

    def test_consolidation_improves_over_fragment(self) -> None:
        image, gt, proposals = self._synthetic_scene()

        cfg = ConsolidationConfig(
            min_area=20,
            descriptor_mode="handcrafted",
            merge=MergeConfig(merge_threshold=0.35, w_texture=0.70, w_boundary=0.30, w_hetero=0.25),
            objective_lambda=0.35,
            objective_mu=0.20,
        )
        cons = TextureSAMV2Consolidator(cfg)
        pred, _ = cons(image, proposals)

        baseline = max(proposals, key=lambda m: int(m.sum()))
        self.assertGreater(iou(pred, gt), iou(baseline, gt) + 0.15)
        self.assertGreater(iou(pred, gt), 0.80)

    def test_empty_input(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cons = TextureSAMV2Consolidator(ConsolidationConfig())
        pred, dbg = cons(image, [])
        self.assertEqual(int(pred.sum()), 0)
        self.assertEqual(dbg.num_input_proposals, 0)


if __name__ == "__main__":
    unittest.main()
