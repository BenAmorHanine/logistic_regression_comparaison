import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Analysis_Framework = pd.read_csv("../input/review/diabetes.csv")
dataset_diabetes = pd.DataFrame(Analysis_Framework)

dataset_diabetes.head(15)

# Load and preprocess dataset
def load_dataset():
    # Load the diabetes dataset
    dataset_diabetes = pd.read_csv("./diabetes_dataset.csv")
    print("First 15 rows of the dataset:")
    print(dataset_diabetes.head(15))

    # Assuming the target column is named 'Outcome' and features are all other columns
    X = dataset_diabetes.drop(columns='Outcome').values
    y = dataset_diabetes['Outcome'].values
    
    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=0.5, random_state=42)