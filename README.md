# Logistic Regression From Scratch

This project implements a complete logistic regression classifier from scratch using pure Python and NumPy. The model is trained and validated on Airbnb host data to predict "superhost" status. 

## Features

- Complete Mathematical Implementation: Custom gradient descent with Newton-Raphson optimization
- Gradient & Hessian Computation: From-scratch implementation of log-loss derivatives
- Convergence Criteria: Automatic stopping based on weight convergence tolerance
- Performance Benchmarking: Speed comparison with scikit-learn's implementation

## Requirements
- Python 3.6+
- NumPy
- Pandas
- Scikit-learn (for comparison)
- Jupyter Notebook

## Installation

1. Clone the repository:
`git clone https://github.com/yourusername/ml-building-logistic-regression.git`

2. Navigate to the project directory:
`cd logistic-regression`

3. Create a virtual environment (recommended):
`python -m venv logistic_env`
`source logistic_env/bin/activate  # On Windows: logistic_env\Scripts\activate`

4. Install required dependencies:
`pip install numpy pandas scikit-learn jupyter`

5. Ensure you have the dataset:
Place airbnbData_train.csv in a `data/` folder within the project directory or modify the file path in the notebook accordingly

## How to Use

Option 1: Run the Jupyter Notebook
1. Start Jupyter Notebook:
`jupyter notebook`

2. Open the notebook:
- Navigate to `LogisticRegressionFromScratch.ipynb` in your browser
- Run all cells sequentially to see the complete workflow

Option 2: Use the Class Directly
```
import numpy as np
import pandas as pd
from LogisticRegressionScratch import LogisticRegressionScratch

# Load your data
X = your_feature_matrix  # Shape: (n_samples, n_features)
y = your_labels         # Shape: (n_samples,) with 0/1 values

# Create and train the model
model = LogisticRegressionScratch(tolerance=1e-8, max_iterations=20)
model.fit(X, y)

# Get results
weights = model.get_weights()
intercept = model.get_intercept()
print(f"Weights: {weights}")
print(f"Intercept: {intercept}")
```

Option 3: Extract and Use as Python Module

1. Convert notebook to Python script:
`jupyter nbconvert --to python LogisticRegressionFromScratch.ipynb`

2. Import and use the class:
`from LogisticRegressionFromScratch import LogisticRegressionScratch`
