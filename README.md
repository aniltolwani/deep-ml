# deep-ml

My solutions to the problems (here)[https://www.deep-ml.com/]

## Problems to solve

1. **K-Nearest Neighbors (KNN) Classifier**
   - Main: Implement a KNN classifier from scratch
   - Extensions:
     a) Optimize for large datasets (e.g., using k-d trees or ball trees)
     b) Implement weighted KNN
     c) Handle multi-class classification
     d) Discuss the curse of dimensionality and its impact on KNN

2. **K-Means Clustering**
   - Main: Implement the k-means clustering algorithm
   - Extensions:
     a) Determine the optimal number of clusters (e.g., elbow method, silhouette score)
     b) Handle initial centroid selection (e.g., k-means++)
     c) Implement a mini-batch version for large datasets
     d) Discuss limitations and when to use alternative clustering methods

3. **Gradient Descent for Linear Regression**
   - Main: Implement gradient descent for linear regression
   - Extensions:
     a) Extend to stochastic gradient descent and mini-batch gradient descent
     b) Implement regularization (L1 and L2)
     c) Adapt for logistic regression
     d) Discuss convergence criteria and learning rate selection

4. **Principal Component Analysis (PCA)**
   - Main: Implement PCA from scratch
   - Extensions:
     a) Determine the optimal number of components to retain
     b) Implement incremental PCA for large datasets
     c) Discuss the relationship between PCA and SVD
     d) Compare PCA with other dimensionality reduction techniques (e.g., t-SNE)

5. **TF-IDF Vectorization**
   - Main: Implement a TF-IDF vectorizer
   - Extensions:
     a) Handle out-of-vocabulary words
     b) Implement n-gram features
     c) Apply sublinear tf scaling
     d) Discuss alternatives like word embeddings (Word2Vec, GloVe)

6. **Data Preprocessing Pipeline**
   - Main: Create a basic data preprocessing pipeline (handling missing values, scaling, encoding)
   - Extensions:
     a) Implement different strategies for handling missing data
     b) Create custom transformers for sklearn Pipeline
     c) Handle mixed data types (numerical and categorical)
     d) Implement feature selection methods

7. **Decision Tree Classifier**
   - Main: Implement a decision tree classifier
   - Extensions:
     a) Handle continuous features (various splitting criteria)
     b) Implement pruning techniques
     c) Extend to Random Forest
     d) Discuss ensemble methods and boosting

8. **Cross-Validation and Model Evaluation**
   - Main: Implement k-fold cross-validation
   - Extensions:
     a) Implement stratified k-fold for imbalanced datasets
     b) Create functions for various evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
     c) Implement learning curves to diagnose bias-variance tradeoff
     d) Discuss cross-validation strategies for time series data

For each problem:
1. Implement the core algorithm in clean, efficient Python code
2. Be prepared to explain the underlying mathematical concepts
3. Discuss time and space complexity
4. Consider how to integrate into a larger ML pipeline
5. Think about real-world applications, especially in the context of search and question-answering systems

