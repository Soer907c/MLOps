import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 1 + np.random.randn(100, 1)

model = LinearRegression()
model.fit(x, y)

predictions = model.predict(x)
mse = mean_squared_error(y, predictions)

print("Training completed successfully")
print(f"MSE: {mse:.4f}")