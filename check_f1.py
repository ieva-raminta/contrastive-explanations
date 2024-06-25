import os
import sys
import re
import numpy as np
import pandas as pd

counter = 0

# iterate through all files starting with 3e- in cwd
for filename in os.listdir(os.getcwd()):
    if filename.startswith('3e-'):
        # read the file
        with open(filename, 'r') as file:
            data = file.read()
        print(filename)
        # get the full line that starts with "test_f1_2"
        lines = data.split('\n')
        for line in lines:
            if 'test_f1_2' in line:
                print(line)
                counter += 1
                print(counter)
                break
