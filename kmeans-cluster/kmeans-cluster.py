import numpy as np

class KNN_Cluster():
    def __init__(self, k=5):
        self.k = k
    
    def l2_distance(self, x, y):
        return np.linalg.norm(np.array(x) - np.array(y))
    
    def fit(self, X):
        self.data = X
        self.n = X.shape[0]
        self.feat = X.shape[1]

        # Random initialization
        self.centroids = np.random.randn(self.k, self.feat)
        self.fit_iteration(100)

    def fit_iteration(self, num_iter):
        for _ in range(num_iter):
            # Calculate pairwise distances for each point
            labels = []
            dists = []
            for idx in range(self.n):
                pairwise_distances = [(i, self.l2_distance(self.data[idx], c)) for i, c in enumerate(self.centroids)]
                sorted_distances = sorted(pairwise_distances, key=lambda x: x[1])  # Remove reverse=True
                labels.append(sorted_distances[0][0])
                dists.append(sorted_distances[0][1])
            
            print(f"iteration: {_}; average distance: {np.mean(dists)}")

            # Recalculate the centroids based on the arithmetic mean in each cluster
            new_centroids = []
            for cluster in range(self.k):
                idxs = [i for i, label in enumerate(labels) if label == cluster]
                if idxs:  # Check if the cluster is not empty
                    average = np.mean(self.data[idxs], axis=0)
                    new_centroids.append(average)
                else:
                    # If a cluster is empty, keep its old centroid
                    new_centroids.append(self.centroids[cluster])
            
            self.centroids = new_centroids

    def predict(self, X):
        labels = []
        for point in X:
            distances = [self.l2_distance(point, c) for c in self.centroids]
            labels.append(np.argmin(distances))
        return labels