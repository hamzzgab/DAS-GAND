import unittest

from src.features import module_1 as calc


class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        :return: None
        """
        data = [1, 2, 3]
        result = calc.sum(data)
        self.assertEqual(result, 6)


if __name__ == '__main__':
    unittest.main()