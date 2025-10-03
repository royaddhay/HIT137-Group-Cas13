# Encapsulation
class OOPExplanation:
    def __init__(self):
        self.__details = {}   # private variable

    def add_detail(self, concept, explanation):
        self.__details[concept] = explanation

    def get_details(self):
        return self.__details

# Multiple Inheritance
class A:
    def greet(self):
        return "Hello from A"

class B:
    def greet(self):
        return "Hello from B"

class C(A, B):   # Multiple inheritance
    def greet(self):
        return super().greet() + " and overridden in C"
