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

    def test_complex_shapes(self):
        '''
        Complex Shapes:

        Expected Behavior: The function should accurately identify and match the complex shapes, providing reasonable match grades and angles. 
        Since the shapes are well-defined and non-overlapping, the function should return a match grade close to 1.0 and angles close to 0.0 for each shape.
        '''
        maska = np.zeros((50, 50), dtype=np.uint8)
        maska_ref = np.zeros((50, 50), dtype=np.uint8)
        maska[5:15, 5:15] = 255
        maska[25:35, 25:35] = 255
        maska_ref[5:15, 5:15] = 255
        maska_ref[25:35, 25:35] = 255
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 2)
        self.assertAlmostEqual(grades[0], 1.0)  # Predicted: 1.0
        self.assertAlmostEqual(angles[0], 0.0)  # Predicted: 0.0
        # Assert expected behavior: Two contours with perfect grades and angles

    def test_masks_with_holes(self):
        '''
        Expected Behavior: The function should handle internal contours correctly, returning only significant external contours with high match grades. 
        Since the internal contours are explicitly defined, the function should recognize and handle them appropriately, returning contours representing the external shapes with match
        grades close to 1.0.
        '''
        maska = np.zeros((50, 50), dtype=np.uint8)
        maska_ref = np.zeros((50, 50), dtype=np.uint8)
        maska[5:15, 5:15] = 255
        maska[8:12, 8:12] = 0
        maska_ref[5:15, 5:15] = 255
        maska_ref[8:12, 8:12] = 0
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 1)
        self.assertAlmostEqual(grades[0], 1.0)  # Predicted: 1.0
        self.assertAlmostEqual(angles[0], 0.0)  # Predicted: 0.0
        # Assert expected behavior: One contour with perfect grade and angle of 0.0

    def test_noise_in_masks(self):
        '''
        Expected Behavior: The function should filter out noise and return only significant contours with high match grades.
        Since the noise is randomly scattered, the function should identify and ignore the noise,
        returning contours corresponding to the significant shapes with match grades close to 1.0.

        '''
          
        maska = np.zeros((50, 50), dtype=np.uint8)
        maska_ref = np.zeros((50, 50), dtype=np.uint8)
        np.random.seed(42)
        noise_indices = np.random.choice(range(50), size=50, replace=True)
        maska[noise_indices, :] = 255
        np.random.seed(24)
        noise_indices = np.random.choice(range(50), size=50, replace=True)
        maska_ref[:, noise_indices] = 255
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 0)
        self.assertEqual(len(grades), 0)
        self.assertEqual(len(angles), 0)
        # Assert expected behavior: No contours, grades, or angles

    def test_border_cases(self):
        '''
        Expected Behavior: The function should handle edge cases gracefully, providing accurate match grades and angles despite the proximity to the border. 
        Since the shapes are close to the border, the function should correctly identify and match them, returning match grades close to 1.0 and angles close to 0.0.
        '''
        maska = np.zeros((50, 50), dtype=np.uint8)
        maska_ref = np.zeros((50, 50), dtype=np.uint8)
        maska[0, :] = 255
        maska_ref[:, -1] = 255
        contours, grades, angles = poisci_ujemajoce_oblike(maska, maska_ref)
        self.assertEqual(len(contours), 2)
        self.assertAlmostEqual(grades[0], 1.0)  # Predicted: 1.0
        self.assertAlmostEqual(angles[0], 0.0)  # Predicted: 0.0
        # Assert expected behavior: Two contours with perfect grades and angles

if __name__ == '__main__':
    unittest.main()