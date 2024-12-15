import numpy as np
import sklearn; 
sklearn.show_versions()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def f(x):
    return 2 * np.log(x + 1) + 3

def generate_data(a, n_samples=1000, noise_std=0.1):
    x = np.random.uniform(-0.05, a, n_samples)
    y = f(x) + np.random.normal(0, noise_std, n_samples)
    return x, y

def perform_regression(x, y):
    model = LinearRegression()
    x_reshaped = x.reshape(-1, 1)
    model.fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    mse = mean_squared_error(y, y_pred)
    return model, mse

def plot_results(x, y, model, a):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label='Data')
    x_line = np.linspace(-0.05, a, 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, color='red', label='Linear Regression')
    plt.plot(x_line, f(x_line), color='green', label='True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Linear Regression for a = {a}')
    plt.legend()
    plt.savefig('my_plot1.png')
    plt.show()

# Case 1: a = 0.01
a1 = 0.01
x1, y1 = generate_data(a1)
model1, mse1 = perform_regression(x1, y1)
plot_results(x1, y1, model1, a1)
print(f"MSE for a = {a1}: {mse1}")

# Case 2: a = 10
a2 = 10
x2, y2 = generate_data(a2)
model2, mse2 = perform_regression(x2, y2)
plot_results(x2, y2, model2, a2)
print(f"MSE for a = {a2}: {mse2}")