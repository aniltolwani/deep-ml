import numpy as np 
def pca(data: np.ndarray, k: int) -> list[list[int|float]]:
	n, m = data.shape
	data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
	cov = np.cov(data_standardized.T)
	eigenvalue, eigenvector = np.linalg.eig(cov)
	# sort the eigenvectors by the eigenvalues
	# eigenvalues is a [1,3] matrix
	idx = np.argsort(eigenvalue.flatten())[::-1]
	sorted_eigenvectors = eigenvector[:,idx]
	principal_components = sorted_eigenvectors[:,:k]
	return np.round(principal_components, 4).tolist()
