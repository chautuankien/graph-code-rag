# Test cases for Phase 4.1: Function/Method <-> Method/Function

# Case 2: Function -> Method
class Calculator:
    def add(self, a, b):
        return a + b

def process_data():
    calc = Calculator()
    calc.add(1, 2)  # should create CALLS edge: Function -> Method

# Case 1: Function -> Function
def helper_func():
    pass

def main_func():
    helper_func()  # should create CALLS edge: Function -> Function

# Case 3: Method -> Function
def log(msg):
    print(msg)

class Logger:
    def save(self):
        log("saving")  # should create CALLS edge: Method -> Function

# Case 4: Method -> Method
class Processor:
    def step1(self):
        self.step2()  # should create CALLS edge: Method -> Method

    def step2(self):
        pass