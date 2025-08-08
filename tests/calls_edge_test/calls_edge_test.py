# import requests
# import numpy as np
# from typing import List

# def helper_func(data):
#     """A helper function that processes data."""
#     print(f"Processing: {data}")
#     return len(data)

# def main_func():
#     """Main function demonstrating various call types."""
#     # Function -> Function
#     result = helper_func("test data")
    
#     # Function -> Built-in
#     print("Hello World")
#     total = sum([1, 2, 3, 4])
#     length = len("hello")
    
#     # Function -> External
#     response = requests.get("https://api.example.com")
#     array = np.array([1, 2, 3])
    
#     # Function -> Constructor
#     user = User("John", "john@example.com")
    
#     return result

# class User:
#     def __init__(self, name: str, email: str):
#         self.name = name
#         self.email = email
        
#     def save(self):
#         """Method -> Built-in"""
#         print(f"Saving user: {self.name}")
        
#         """Method -> Method (same class)"""
#         self.validate()
        
#         """Method -> Function"""
#         helper_func(self.name)
        
#     def validate(self):
#         """Method -> Built-in"""
#         if len(self.email) > 0:
#             return True
#         return False
    
#     def get_profile(self):
#         """Method -> Constructor"""
#         profile = Profile(self.name)
#         return profile

# class Profile:
#     def __init__(self, name: str):
#         self.name = name

# def nested_example():
#     """Function with nested function."""
    
#     def inner_func():
#         """Nested function -> Built-in"""
#         print("Inside nested function")
        
#         """Nested function -> Function (outer scope)"""
#         helper_func("nested call")
    
#     # Function -> Nested function
#     inner_func()