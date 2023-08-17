import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os 

class Data_collection():

    def validate_path(self, user_path):

        # Check if the path exist in the same file system as my script. 
        if os.path.exists(user_path):
            # Check if it is an absolute path. 
            if os.path.isabs(user_path):
                # try the path and raise error message if it is not readable. 
                try:
                    with open (user_path, "r") as file:
                        df = pd.read_csv(file)
                        return df 
                # catch all the exceptions and assign it to variable exc       
                except Exception as exc:
                    print(f"Error reading file {str(exc)}")
            # If it is an absolute path, we will convert it. 
            if not os.path.isabs(user_path):   
                # Convert to absolut path by creating a new path based\
                #  on current working directory. 
                data_path = Path.cwd() / user_path
                # try the path and raise error if it is not readable. 
                try:
                    with open (data_path, "r") as file:
                        df = pd.read_csv(file)
                        return df 
                except Exception as exc:
                    print(f"Error reading file {str(exc)}")

        raise FileNotFoundError("Path does not exist, please check and enter again!")

