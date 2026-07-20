import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.artifact_reliability_benchmark import (
    effective_modes,
    extract_digit_grid,
    normalize_digit,
    pix2pix_pair_metric,
)


class ArtifactReliabilityTests(unittest.TestCase):
    def test_normalize_digit_has_stable_shape_and_range(self):
        image = np.zeros((24, 18), dtype=float)
        image[4:20, 7:11] = 1.0
        normalized = normalize_digit(image)
        self.assertEqual(normalized.shape, (16, 16))
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_extract_digit_grid_returns_forty_tiles(self):
        image = np.zeros((100, 160), dtype=np.uint8)
        for row in range(5):
            for column in range(8):
                image[row * 20 + 5 : row * 20 + 15, column * 20 + 8 : column * 20 + 12] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.png"
            Image.fromarray(image).save(path)
            tiles = extract_digit_grid(path)
        self.assertEqual(tiles.shape, (40, 16, 16))

    def test_effective_modes_is_ten_for_uniform_occupancy(self):
        modes, entropy = effective_modes(np.tile(np.arange(10), 4), 10)
        self.assertAlmostEqual(modes, 10.0)
        self.assertAlmostEqual(entropy, 1.0)

    def test_pix2pix_metric_detects_background_color_spill(self):
        source = np.full((64, 64, 3), 255, dtype=np.uint8)
        source[28:36, 10:54] = 0
        output = source.copy()
        output[:20, :20] = np.asarray([255, 0, 0], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "image_or0.png"
            output_path = root / "image_0.png"
            Image.fromarray(source).save(source_path)
            Image.fromarray(output).save(output_path)
            metric = pix2pix_pair_metric(source_path, output_path)
        self.assertGreater(metric.background_chroma_spill, 0.05)


if __name__ == "__main__":
    unittest.main()
