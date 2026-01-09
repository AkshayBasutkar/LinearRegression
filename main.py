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
    
    

    def anova(self, x, y):
        n = len(y)
        y_pred = self.predict(x)
        
        y_mean = sum(y) / n

        sst = 0.0
        sse = 0.0 
        ssr = 0.0

        for i in range(n):
            sst += np.power((y[i] - y_mean), 2)
            ssr += np.power((y_pred[i] - y_mean), 2)


        self.sse = sst - ssr
        self.sst = sst
        self.ssr = ssr

        self.msr = self.ssr
        self.mse = self.sse / (n - 1)

        self.f_stat = self.msr / self.mse

        print(f"""
ANOVA TABLE 
Source     \tSqaured Sum  \t Mean Squared Sum \t F-Statistic
Regression  {self.ssr}            {self.msr}               {self.f_stat} 
Residual    {self.sse}            {self.mse}
Total       {self.sst}
              """)
        
    
    def score(self, x, y):
        y_pred = self.predict(x)
        y_mean = sum(y) / len(y)

        sse = 0.0
        sst = 0.0

        for i in range(len(y)):
            sse += (y[i] - y_pred[i]) ** 2
            sst += (y[i] - y_mean) ** 2

        return 1 - (sse / sst)  

    def adj_r2(self, x, y):
        n = len(y)
        p = 1
        r2 = self.score(x, y)

        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adj_r2