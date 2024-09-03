import math
import numpy as np
from collections import Counter, defaultdict
import random
import heapq

class KNN():
    def __init__(self):
        self.feature_vectors = []
        self.classes = []

    def add_data(self, batch):
        for X, y in batch:
            self.feature_vectors.append(X)
            self.classes.append(y)
    
    def l2_distance(self, x: list[float], y: list[float]):
        return np.linalg.norm(np.array(x) - np.array(y))

    def predict(self, test: list[float], k: int = 5):
        if len(test) != len(self.feature_vectors[0]):
            raise ValueError("Wrong length")
        distances = [self.l2_distance(test, feat) for feat in self.feature_vectors]
        top_k_indices = [idx for _, idx in sorted((val, idx) for idx, val in enumerate(distances))[:k]]
        classes = [self.classes[idx] for idx in top_k_indices]
        return Counter(classes).most_common(1)[0][0]

class LSH_KNN():
    def __init__(self, num_hashes=10, num_bands=5):
        self.feature_vectors = []
        self.classes = []
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]
        self.hash_functions = []

    def _generate_hash_function(self):
        if not self.feature_vectors:
            return lambda x: 0
        a = np.random.randn(len(self.feature_vectors[0]))
        b = random.uniform(0, 1)
        return lambda x: int(np.dot(a, x) + b)

    def _hash_vector(self, vector):
        return [hash_fn(vector) for hash_fn in self.hash_functions]

    def add_data(self, batch):
        if not self.hash_functions:
            self.hash_functions = [self._generate_hash_function() for _ in range(self.num_hashes)]
        for X, y in batch:
            self.feature_vectors.append(X)
            self.classes.append(y)
            hash_values = self._hash_vector(X)
            for i in range(self.num_bands):
                band_hash = tuple(hash_values[i::self.num_bands])
                self.hash_tables[i][band_hash].append(len(self.feature_vectors) - 1)

    def l2_distance(self, x, y):
        return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))

    def predict(self, test, k=5):
        hash_values = self._hash_vector(test)
        candidate_indices = set()
        for i in range(self.num_bands):
            band_hash = tuple(hash_values[i::self.num_bands])
            candidate_indices.update(self.hash_tables[i][band_hash])
        
        distances = [(self.l2_distance(test, self.feature_vectors[idx]), idx) for idx in candidate_indices]
        top_k_indices = [idx for _, idx in sorted(distances)[:k]]
        top_k_classes = [self.classes[idx] for idx in top_k_indices]
        return Counter(top_k_classes).most_common(1)[0][0]

class KDTreeNode:
    def __init__(self, point, label, left=None, right=None):
        self.point = point
        self.label = label
        self.left = left
        self.right = right

class KDTree_KNN():
    def __init__(self):
        self.root = None

    def add_data(self, batch):
        points = [(X, y) for X, y in batch]
        self.root = self._build_tree(points, depth=0)

    def _build_tree(self, points, depth):
        if not points:
            return None
        k = len(points[0][0])
        axis = depth % k
        points.sort(key=lambda x: x[0][axis])
        median = len(points) // 2
        return KDTreeNode(
            point=points[median][0],
            label=points[median][1],
            left=self._build_tree(points[:median], depth + 1),
            right=self._build_tree(points[median + 1:], depth + 1)
        )

    def l2_distance(self, x, y):
        return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))

    def _knn_search(self, node, point, k, depth, heap):
        if node is None:
            return
        axis = depth % len(point)
        dist = self.l2_distance(point, node.point)
        if len(heap) < k or dist < -heap[0][0]:
            heapq.heappush(heap, (-dist, node.label))
            if len(heap) > k:
                heapq.heappop(heap)
        diff = point[axis] - node.point[axis]
        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._knn_search(close, point, k, depth + 1, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn_search(away, point, k, depth + 1, heap)

    def predict(self, test, k=5):
        heap = []
        self._knn_search(self.root, test, k, 0, heap)
        top_k_classes = [label for _, label in heap]
        return Counter(top_k_classes).most_common(1)[0][0]

## fake data

X = np.random.randn(10000, 1000)
y = np.random.randint(low=0, high=2, size=(10000,))
batch = list(zip(X, y))
toy_knn = KNN()
toy_knn.add_data(batch)

# time the differences between the three models

import time

def time_model(model, test_data, k=5):
    start_time = time.time()
    for test in test_data:
        model.predict(test, k)  
    end_time = time.time()
    return end_time - start_time

test_data = np.random.randn(1000, 1000)

# Initialize models with data
toy_lsh_knn = LSH_KNN()
toy_lsh_knn.add_data(batch)

toy_kdtree_knn = KDTree_KNN()
toy_kdtree_knn.add_data(batch)

print("KNN: ", time_model(toy_knn, test_data))
print("LSH KNN: ", time_model(toy_lsh_knn, test_data))
print("KDTree KNN: ", time_model(toy_kdtree_knn, test_data))

