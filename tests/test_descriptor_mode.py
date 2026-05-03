import unittest

from texturesam_v2.consolidator import ConsolidationConfig, TextureSAMV2Consolidator


class TestDescriptorMode(unittest.TestCase):
    def test_dtd_mode_requires_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            TextureSAMV2Consolidator(ConsolidationConfig(descriptor_mode="dtd_cnn", dtd_checkpoint=None))


if __name__ == "__main__":
    unittest.main()
