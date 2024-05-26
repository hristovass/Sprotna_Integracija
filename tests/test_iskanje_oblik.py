import numpy as np
import unittest
from iskanje_oblik import discretizeConture, normaliziraj_konturo, poisci_ujemajoce_oblike, primerjaj_konturi
import unittest

class TestNormalizeContour(unittest.TestCase):
    def test_normalization(self):
        # Test if normalization works as expected
        contour = np.array([[0, 0], [1, 1], [2, 2]])
        expected_normalized_contour = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        normalized_contour = normaliziraj_konturo(contour)
        np.testing.assert_array_almost_equal(normalized_contour, expected_normalized_contour)

    def test_zero_input(self):
        # Test if function handles zero input gracefully
        contour = np.zeros((0, 2))
        normalized_contour = normaliziraj_konturo(contour)
        self.assertEqual(len(normalized_contour), 0)

    def test_single_point(self):
        # Test if function handles a single point gracefully
        contour = np.array([[1, 1]])
        normalized_contour = normaliziraj_konturo(contour)
        expected_normalized_contour = np.array([[0.0, 0.0]])
        np.testing.assert_array_almost_equal(normalized_contour, expected_normalized_contour, decimal=5)

class TestPrimerjajKonturiEdgeCases(unittest.TestCase):
    def setUp(self):
        # Define sample contours for testing
        self.kontura_ref = [(0, 0), (0, 5), (5, 5), (5, 0)]
        self.kontura_primer = [(1, 1), (1, 4), (4, 4), (4, 1)]

    def test_best_angle(self):
        # Test if the best angle returned is within a reasonable range
        grade, best_angle = primerjaj_konturi(self.kontura_ref, self.kontura_primer)
        self.assertTrue(-2*np.pi <= best_angle <= 2*np.pi)

    def test_similarity_score(self):
        # Test if the function returns a similarity score between 0 and 1
        grade, best_angle = primerjaj_konturi(self.kontura_ref, self.kontura_primer)
        self.assertTrue(0 <= grade <= 1)

    def test_identical_contours(self):
        # Test when the input contours are identical
        c_1 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        c_2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        iou_expected = 1.0
        angle_ref = 0.0

        iou_est, angle = primerjaj_konturi(c_1, c_2, locljivost=100, koraki=4)

        self.assertAlmostEqual(iou_est, iou_expected, places=5)
        self.assertAlmostEqual(angle, angle_ref, places=5)

    def test_rotation_invariance(self):
        # Test if the function returns the same similarity score for rotated versions of the same contour
        grade1, _ = primerjaj_konturi(self.kontura_ref, self.kontura_primer)
        grade2, _ = primerjaj_konturi(self.kontura_ref, np.roll(self.kontura_primer, 1, axis=0))
        self.assertEqual(grade1, grade2)
    
    def test_large_resolution(self):
        # Test with very large resolution
        c_1 = np.array([[0, 0], [0, 5], [5, 5], [5, 0]])
        c_2 = np.array([[1, 1], [1, 4], [4, 4], [4, 1]])
        iou_est, angle = primerjaj_konturi(c_1, c_2, locljivost=500)
        self.assertTrue(0 <= iou_est <= 1)
        self.assertTrue(-2*np.pi <= angle <= 2*np.pi)

class TestShapeMatching(unittest.TestCase):
    
    def test_empty_masks(self):
        maska = np.zeros((10, 10), dtype=np.uint8)
        maska_ref = np.zeros((10, 10), dtype=np.uint8)
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 0)
        self.assertEqual(len(grades), 0)
        self.assertEqual(len(angles), 0)

    def test_large_masks_no_matching_contours(self):
        maska = np.zeros((100, 100), dtype=np.uint8)
        maska_ref = np.zeros((100, 100), dtype=np.uint8)
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 0)
        self.assertEqual(len(grades), 0)
        self.assertEqual(len(angles), 0)

    def test_identical_masks(self):
        maska = np.ones((50, 50), dtype=np.uint8) * 255
        maska_ref = np.ones((50, 50), dtype=np.uint8) * 255
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 1)
        self.assertAlmostEqual(grades[0], 1.0)
        self.assertAlmostEqual(angles[0], 0.0)

    def test_nearly_identical_masks(self):
        maska = np.zeros((50, 50), dtype=np.uint8)
        maska_ref = np.zeros((50, 50), dtype=np.uint8)
        maska[20:30, 20:30] = 255
        maska_ref[21:31, 21:31] = 255
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 1)
        self.assertAlmostEqual(grades[0], 1.0)  # Predicted: Close to 1.0
        self.assertAlmostEqual(angles[0], 0.0)  # Predicted: Close to 0.0
        # Assert expected behavior: One contour with high grade and minimal angle

if __name__ == '__main__':
    unittest.main()