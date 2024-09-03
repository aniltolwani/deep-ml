import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	m, n = X.shape
	theta = np.zeros((n, 1))
	y = y.reshape((m,1))
	
	for _ in range(iterations):
		y_bar = X @ theta
		error = y_bar - y
		gradient = (1/m) * (X.T @ error)
		theta = theta - (alpha * gradient)
	return theta