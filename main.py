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

        self.slope = numerator / denominator

        self.intercept = y_mean - (self.slope * x_mean)
        self.is_fitted = True

    
    def predict(self, x):
        if not self.is_fitted:
            print("model not fitted on dataset")
            return
        if isinstance(x, (list, tuple, np.ndarray)):
            return [self.intercept + (self.slope * x_val) for x_val in x]
        else:
            return self.intercept + (self.slope * x)
    
    def MSE(self, x, y):
        if not self.is_fitted:
            print("Model is not fitted")
            return
        
        y_pred = self.predict(x)

        total_error = 0
        for i in range(len(y_pred)):
            total_error += (y[i] - y_pred[i]) ** 2

        mse = total_error / len(y)

        return mse



X = np.linspace(1, 10, 10)
y = 4 * X + 2

model = SimpleLinearRegression()
model.fit(X, y)

print(model.predict([1, 2, 3]))
print(model.MSE(X, y))

