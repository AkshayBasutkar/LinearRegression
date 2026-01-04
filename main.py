import numpy as np
from sklearn.model_selection import train_test_split

class SimpleLinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
        self.is_fitted = False

    def fit(self, x, y):
        
        n = len(x)

        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        numerator = 0.0
        denominator = 0.0

        for i in range(len(x)):

            input_deviation = x[i] - x_mean
            output_deviation = y[i] - y_mean

            numerator += input_deviation*output_deviation
            denominator += input_deviation ** 2

        intercept = numerator / denominator

        slope = y_mean - (intercept * x_mean)
        self.is_fitted = True

        model.intercept = intercept
        model.slope = slope

        return slope, intercept
    
    def predict(self, x):
        if self.is_fitted:
            return self.intercept + (self.slope * x)
        else:
            print("model not fitted on dataset")
    
    def MSE(self, x, y):
        if not self.is_fitted:
            print("Model is not fitted")
            return
        
        y_pred = []
        for i in range(len(x)):
            y_pred.append(self.predict(x[i]))

        total_error = 0
        for i in range(len(y_pred)):
            error = (y[i] - y_pred[i]) ** 2
            total_error += error

        mse = total_error / len(y)

        return mse


np.random.seed(42) 
X = 2.5 * np.random.rand(100, 1) 
y = 5 + 3 * X + np.random.randn(100, 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SimpleLinearRegression()
model.predict(10)
slope, intercept = model.fit(X_train, y_train)
print(f"Slope: {slope}, intercept: {intercept}")
model.predict(10)
mse = model.MSE(X_test, y_test)
print(f"Mean Squared Error: {mse}")
