import unittest
import numpy as np
from naive_ptq.ptq.quantization import QuantizedArray

class quant_Test(unittest.TestCase):
    def test_quant(self):
        fp1 = np.array([1, 2, 3, 4], dtype=np.float64)
        fp2 = np.array([1, 2.001, 3, 4], dtype=np.float64)
        q1 = QuantizedArray(fp1, 4, 8)
        q2 = QuantizedArray(fp2, 4, 8)
        np.testing.assert_allclose(q1.int_arr, np.array([16, 32, 48, 64], dtype=np.int32))
        np.testing.assert_allclose(q2.int_arr, np.array([16, 32, 48, 64], dtype=np.int32))

    def test_dequant(self):
        fp = np.array([1, 2, 3, 4], dtype=np.float64)
        q = QuantizedArray(fp, 4, 8)
        np.testing.assert_allclose(q.Dequantize(), fp)

    def test_quant_add(self):
        fp1 = np.array([1, 2, 3, 4], dtype=np.float64)
        fp2 = np.array([1, 2.001, 3, 4], dtype=np.float64)
        q1 = QuantizedArray(fp1, 4, 8)
        q2 = QuantizedArray(fp2, 4, 8)
        q_add = q1 + q2
        np.testing.assert_allclose(q_add.int_arr, np.array([32, 64, 96, 128], dtype=np.int32))
        self.assertEqual(q_add.shiftbit, 4)

    def test_quant_mul(self):
        fp1 = np.array([1, 2], dtype=np.float64)
        fp2 = np.array([[1], [2]], dtype=np.float64)
        q1 = QuantizedArray(fp1, 2, 8) # 4, 8
        q2 = QuantizedArray(fp2, 3, 8) # 8, 16
        q_mul = q1 * q2
        np.testing.assert_allclose(q_mul.int_arr, np.array([160], dtype=np.int32))
        self.assertEqual(q_mul.shiftbit, 5)
        np.testing.assert_allclose(q_mul.Dequantize(), np.array([5]))