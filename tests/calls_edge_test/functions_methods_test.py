# Test cases for Phase 4.1: Function/Method <-> Method/Function

import requests
import numpy as np

def foo():
    x = requests.get("http://example.com")    # Function→External, callee_raw="requests.get", callee_type=EXTERNAL
    y = np.array([1,2,3])                     # Function→External, callee_raw="np.array", callee_type=EXTERNAL

class A:
    def bar(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1,2]}) 

def foo():
    print("hello")   # CALLS: Function→Built-in, callee_raw=print, callee_type=BUILTIN

class A:
    def bar(self):
        a = len([1,2,3])

# Case 2: Function -> Method
# class Calculator:
#     def add(self, a, b):
#         return a + b

# def process_data():
#     calc = Calculator()
#     # calc.add(1, 2)  # should create CALLS edge: Function -> Method

# # Case 1: Function -> Function
# def helper_func():
#     pass

# def main_func():
#     helper_func()  # should create CALLS edge: Function -> Function

# # Case 3: Method -> Function
# def log(msg):
#     print(msg)

# class Logger:
#     def save(self):
#         log("saving")  # should create CALLS edge: Method -> Function

# # Case 4: Method -> Method
# class Processor:
#     def step1(self):
#         self.step2()  # should create CALLS edge: Method -> Method

#     def step2(self):
#         pass