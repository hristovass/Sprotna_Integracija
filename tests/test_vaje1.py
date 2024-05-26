import unittest
#from vaje1 import Counter, FibonacciSequence, avg 
from vaje1 import *
import time

class TestCaseAVG(unittest.TestCase):

    def test_hard_input(self):
        with self.assertRaises(TypeError):
            self.assertEqual(avg(1, 2, 3, None), 2)

    def test_hard_input_two(self):
        with self.assertRaises(TypeError):
            self.assertEqual(avg(10, 10, 10, 10, float), 10)

    def test_hard_input_three(self):
        with self.assertRaises(TypeError):
            self.assertEqual(avg(10, 10, 10, 10, frozenset), 10)

    def test_hard_input_four(self):
        with self.assertRaises(TypeError):
            self.assertEqual(avg(10, 10, 10, 10, set), 10)


class TestCaseCounter(unittest.TestCase):
    def setUp(self):
        self.counter = Counter()

    def test_medium_input(self):
        self.counter.add()
        self.counter.add()
        self.counter.add()
        self.assertEqual(self.counter.get_value(), 3)

    def test_medium_input_two(self):
        self.counter.add()
        self.counter.add()
        self.counter.add()
        self.counter.remove()
        self.counter.remove()
        self.assertEqual(self.counter.get_value(), 1)

    def test_hard_input(self):
        self.counter.remove()
        self.counter.remove()
        self.counter.remove()
        self.counter.remove()
        self.assertEqual(self.counter.get_value(), 0)

    def test_hard_input_two(self):
        for _ in range(0, 1000):
            self.counter.add()
        self.assertEqual(self.counter.get_value(), 1000)

    def tearDown(self):
        self.counter = None


class TestCaseTestEfficiency(unittest.TestCase):

    def setUp(self):
        self._fibonacci_sequence = FibonacciSequence()
        self._efficiency_data = dict()

    def test_first_method(self):
        starting_time = time.time()

        self._fibonacci_sequence.recursive_method(20)

        ending_time = time.time()

        self._efficiency_data['recursive_method'] = ending_time - starting_time

    def test_second_method(self):
        starting_time = time.time()

        self._fibonacci_sequence.math_method(20)

        ending_time = time.time()

        self._efficiency_data['math_method'] = ending_time - starting_time

    def tearDown(self):
        print(self._efficiency_data)
        self._fibonacci_sequence = None
        self._efficiency_data.clear()
        
class TestCaseTestCart(unittest.TestCase):
    def setUp(self):
        # Create product objects for testing
        self.item1 = Item('phone', 1000)
        self.item2 = Item('laptop', 1500)
        self.item3 = Item('tv', 2000)

        # Create an object of class Cart for testing
        self.cart = Cart()

    def test_add_item(self):
        # Add an item to the cart and check if it is in the list of items in the cart
        self.cart.add_item(self.item1)
        self.assertIn(self.item1, self.cart.get_items())

    def test_remove_item(self):
        # Add an item to the cart, then delete it and check if it is not in the list of items in the cart
        self.cart.add_item(self.item1)
        self.cart.remove_item(self.item1)
        self.assertNotIn(self.item1, self.cart.get_items())

    def test_get_total_price(self):
        # Add multiple items to the cart and check if the total cost is calculated correctly
        self.cart.add_item(self.item1)
        self.cart.add_item(self.item2)
        self.cart.add_item(self.item3)
        expected_total_price = self.item1.get_price() + self.item2.get_price() + self.item3.get_price()
        self.assertEqual(self.cart.get_total_price(), expected_total_price)

    def test_clear(self):
        # Add an item to the cart, then empty the cart and check if it's empty
        self.cart.add_item(self.item1)
        self.cart.clear()
        self.assertEqual(len(self.cart.get_items()), 0)

if __name__ == '__main__':
    unittest.main()