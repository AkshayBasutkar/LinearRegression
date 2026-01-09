# Simple Linear Regression from Scratch

A pure Python implementation of Simple Linear Regression built from scratch without using high-level machine learning libraries. This project demonstrates the mathematical foundations of linear regression and provides a complete, educational implementation with visualization and statistical analysis features.

## Overview

This implementation calculates regression coefficients using the ordinary least squares (OLS) method, computing the slope and intercept from first principles. It includes comprehensive statistical metrics and visualization capabilities to analyze model performance.

## Features

- **Pure Implementation**: Built from scratch using only NumPy for numerical operations
- **Complete API**: Familiar scikit-learn-like interface with `fit()` and `predict()` methods
- **Statistical Metrics**:
  - R² (Coefficient of Determination)
  - Adjusted R²
  - Mean Squared Error (MSE)
  - ANOVA table with F-statistic
- **Visualization**:
  - Regression line plot
  - Residual plot for model diagnostics
- **Model Summary**: Comprehensive summary statistics in a formatted table
- **Validation**: Tested against scikit-learn for accuracy verification

## Requirements

```
numpy
scikit-learn
matplotlib
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AkshayBasutkar/LinearRegression.git
cd LinearRegression
```

2. Install required packages:
```bash
pip install numpy scikit-learn matplotlib
```

## Usage

### Basic Example

```python
from simpleLinearRegression import SimpleLinearRegression
import numpy as np

# Create sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.3, 4.1, 5.8, 8.2, 10.1, 11.9, 14.2, 16.0, 17.8, 20.1])

# Create and train model
model = SimpleLinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict([11, 12, 13])
print(f"Predictions: {predictions}")

# Get model summary
model.summary(X, y)
```

### Advanced Usage

```python
from simpleLinearRegression import SimpleLinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load dataset
data = load_diabetes()
X = data.data[:, 2]  # BMI feature
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = SimpleLinearRegression()
model.fit(X_train, y_train)

# Evaluate model
print(f"R² Score: {model.score(X_test, y_test):.4f}")
print(f"MSE: {model.MSE(X_test, y_test):.4f}")
print(f"Adjusted R²: {model.adj_r2(X_test, y_test):.4f}")

# Display comprehensive summary
model.summary(X_test, y_test)

# Show ANOVA table
model.anova(X_test, y_test)

# Visualize results
model.plot_regression(X_test, y_test)
model.plot_residuals(X_test, y_test)
```

## API Reference

### `SimpleLinearRegression()`

Creates a new Simple Linear Regression model.

#### Methods

##### `fit(x, y)`
Fits the model to training data using the ordinary least squares method.
- **Parameters**:
  - `x`: array-like, independent variable (features)
  - `y`: array-like, dependent variable (target)
- **Returns**: None (updates model parameters in-place)

##### `predict(x)`
Makes predictions using the fitted model.
- **Parameters**:
  - `x`: array-like or single value, data to predict on
- **Returns**: Predictions (list or single value)

##### `score(x, y)`
Calculates the R² (coefficient of determination) score.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: R² score (float)

##### `adj_r2(x, y)`
Calculates the Adjusted R² score.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: Adjusted R² score (float)

##### `MSE(x, y)`
Calculates Mean Squared Error.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: MSE value (float)

##### `summary(x, y)`
Prints a comprehensive summary of model statistics.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: None (prints to console)

##### `anova(x, y)`
Displays ANOVA table with Sum of Squares and F-statistic.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: None (prints to console)

##### `plot_regression(x, y)`
Plots the regression line against actual data points.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: None (displays plot)

##### `plot_residuals(x, y)`
Plots residuals to check model assumptions.
- **Parameters**:
  - `x`: array-like, independent variable
  - `y`: array-like, dependent variable
- **Returns**: None (displays plot)

#### Attributes

- `slope`: The slope coefficient (β₁) of the regression line
- `intercept`: The intercept coefficient (β₀) of the regression line
- `is_fitted`: Boolean indicating whether the model has been fitted

## Testing

The repository includes a validation script (`testing.py`) that compares the implementation against scikit-learn's LinearRegression to ensure accuracy.

Run the validation:
```bash
python testing.py
```

This will:
- Load the diabetes dataset from scikit-learn
- Train both the custom implementation and scikit-learn's model
- Compare slopes, intercepts, MSE, R², and Adjusted R²
- Verify that results match within numerical precision

## Mathematical Background

The simple linear regression model fits a line:

```
y = β₀ + β₁x
```

Where:
- β₀ (intercept) = ȳ - β₁x̄
- β₁ (slope) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]

The implementation calculates these coefficients using the method of ordinary least squares (OLS), which minimizes the sum of squared residuals.

## Project Structure

```
LinearRegression/
│
├── simpleLinearRegression.py   # Main implementation
├── testing.py                   # Validation against scikit-learn
└── readme.md                    # This file
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## License

This project is open source and available for educational purposes.

## Author

[Akshay Basutkar](https://github.com/AkshayBasutkar)

## Acknowledgments

- Inspired by scikit-learn's LinearRegression API
- Built as an educational project to understand the mathematics behind linear regression