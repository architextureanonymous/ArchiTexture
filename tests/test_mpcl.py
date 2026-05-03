import unittest

import torch

from texturesam_v2.mpcl import mpcl_loss


class TestMPCL(unittest.TestCase):
    def test_mpcl_identical_masks_near_zero(self) -> None:
        base = torch.rand(4, 32, 32)
        loss = mpcl_loss(base)
        self.assertGreaterEqual(float(loss.item()), 0.0)

        same = (base[0:1] > 0.5).float().repeat(4, 1, 1)
        loss_same = mpcl_loss(same)
        self.assertLess(float(loss_same.item()), 1e-4)

    def test_mpcl_disjoint_masks_high(self) -> None:
        m1 = torch.zeros(32, 32)
        m2 = torch.zeros(32, 32)
        m1[:16, :16] = 1.0
        m2[16:, 16:] = 1.0
        masks = torch.stack([m1, m2], dim=0)
        loss = mpcl_loss(masks)
        self.assertGreater(float(loss.item()), 0.95)


if __name__ == "__main__":
    unittest.main()
