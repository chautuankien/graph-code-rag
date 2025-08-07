from src.project.subpackage.model import ModelTest
from dataclasses import dataclass

@dataclass
class Point:
    """This is a docstring"""
    x: int
    y: int

# Python code to illustrate
# Decorators basic in Python
def decorator_fun(func):
    print("Inside decorator")
    def inner(*args,**kwargs):
        print("Inside inner function")
        print("Decorated the function")
        # do operations with func
        func()
    return inner()
@decorator_fun
def func_to():
    print("Inside actual function")
    
decorator_fun(func_to)()


point = Point(x=3, y=2)
# Printing object
print(point)

def run(a: int, b: int) -> int:
    c = a + b
    def test(c):
        print("hello")
    return test(c)