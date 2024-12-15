'''
Template for Assignment 1
'''
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.optim as optim


# Function to plot a polynomial
def plot_polynomial(coeffs, z_range, color='b'):
    z_vals = np.linspace(z_range[0], z_range[1], 500)  # Create 500 points in the given range
    poly_vals = np.polyval(coeffs[::-1], z_vals)  # Evaluate polynomial at these points
    plt.plot(z_vals, poly_vals, color=color)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.grid(True)
    plt.savefig('my_plot1.png')
    plt.show()

# Define coefficients and plot the polynomial
coeffs = np.array([1, -1, 5, -0.1, 1/30])
plot_polynomial(coeffs, [-4, 4], color='b')

# Function to create a dataset with noise
def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    np.random.seed(seed)
    z = np.random.uniform(z_range[0], z_range[1], sample_size)  # Generate random z values
    noise = np.random.normal(0, sigma, sample_size)  # Generate Gaussian noise
    y = sum([coeffs[i] * z**i for i in range(len(coeffs))]) + noise  # Create y values with noise
    X = np.array([z**i for i in range(len(coeffs))]).T  # Create feature matrix
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create training and validation datasets
X_train, y_train = create_dataset(coeffs=[1, -1, 5, -0.1, 1/30], z_range=[-2, 2], sample_size=500, sigma=0.5, seed=0)
X_val, y_val = create_dataset(coeffs=[1, -1, 5, -0.1, 1/30], z_range=[-2, 2], sample_size=500, sigma=0.5, seed=1)



# Function to visualize the data and true polynomial
def visualize_data(X, y, coeffs, z_range, title=""):
    z = np.linspace(z_range[0], z_range[1], 100)
    p_z = sum([coeffs[i] * z**i for i in range(len(coeffs))])
    plt.plot(z, p_z, label='True Polynomial', color='r')
    plt.scatter(X[:, 1].numpy(), y.numpy(), label='Noisy Data')
    plt.title(title)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    plt.grid(True)
    plt.savefig('my_plot2.png')
    plt.show()
    
visualize_data(X_train, y_train, coeffs, z_range=[-2, 2], title='Training Data Visualization')
# Define the polynomial regression model
class PolynomialRegressionModel(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.linear = nn.Linear(degree + 1, 1, bias=False)  # Linear layer without bias
    
    def forward(self, x):
        return self.linear(x)

# Create model, loss function, and optimizer
model = PolynomialRegressionModel(degree=4)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Lists to store losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(500):
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    train_losses.append(loss.item())

    # Compute validation loss
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Plot training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('my_plo3t.png')
plt.show()

# Generate predictions using the trained model
with torch.no_grad():
    z = np.linspace(-4, 4, 100)
    X_test = torch.tensor([[1, z_i, z_i**2, z_i**3, z_i**4] for z_i in z], dtype=torch.float32)
    y_pred = model(X_test).numpy().flatten()

# Plot estimated vs true polynomial
plt.plot(z, y_pred, label='Estimated Polynomial', color='b')
plt.plot(z, sum([coeffs[i] * z**i for i in range(len(coeffs))]), label='True Polynomial', color='r')
plt.xlabel('z')
plt.ylabel('p(z)')
plt.title('Estimated vs True Polynomial')
plt.legend()
plt.grid(True)
plt.savefig('my_plot4.png')
plt.show()

# Track parameter values during training
parameter_values = {f'Weight {i}': [] for i in range(len(coeffs))}

for epoch in range(500):
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store current parameter values
    for i, param in enumerate(model.parameters()):
        parameter_values[f'Weight {i}'].append(param.data.numpy().flatten()[0])

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Plot parameter values over time
plt.figure(figsize=(10, 6))
for key, values in parameter_values.items():
    plt.plot(values, label=key)

# Plot true coefficient values
true_coeffs = [1, -1, 5, -0.1, 1/30]
for i, coeff in enumerate(true_coeffs):
    plt.axhline(y=coeff, linestyle='--', label=f'True Value {i}')

plt.title('Parameter Values During Training')
plt.xlabel('Epochs')
plt.ylabel('Parameter Values')
plt.legend()
plt.grid(True)
plt.savefig('my_plot.png')
plt.show()
