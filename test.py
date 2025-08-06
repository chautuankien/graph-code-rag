import os

path = "tests/sample_repo"

for root, dirs, filenames in os.walk(path):
    print("root:", root)
    print("---")
    print(dirs)
    print("---")
    print(filenames)