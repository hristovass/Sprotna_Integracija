from math import sqrt

def avg(*list_numbers: float) -> float:
	total = 0
	for num in list_numbers:
		if isinstance(num, (int, float)):
			total += num
		else:
			raise TypeError("Wrong input data. Please make sure that everything is a number. ")
	return total / len(list_numbers)

class Counter:
    def __init__(self):
        self._value = 0

    def add(self):
        self._value += 1

    def remove(self):
        if self._value <= 0:
            self._value = 0
        else:
            self._value -= 1

    def clear(self):
        self._value = 0

    def get_value(self):
        return self._value

class FibonacciSequence:

    def recursive_method(self, n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return self.recursive_method(n - 1) + self.recursive_method(n - 2)

    def math_method(self, n):
        return ((1 + sqrt(5)) ** n - (1 - sqrt(5)) ** n) / (2 ** n * sqrt(5))
    
class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def get_name(self):
        return self.name

    def get_price(self):
        return self.price


class Cart:
    def __init__(self):
        self.items = []  # List to store items in the cart

    def add_item(self, item):
        self.items.append(item)  # Add an item to the cart

    def remove_item(self, item):
        if item in self.items:
            self.items.remove(item)  # Remove an item from the cart if it exists

    def get_items(self):
        return self.items  # Return the list of items in the cart

    def clear(self):
        self.items = []  # Clear all items from the cart

    def get_total_price(self):
        total_price = 0
        for item in self.items:
            total_price += item.get_price()  # Calculate the total price of all items in the cart
        return total_price  # Return the total price
