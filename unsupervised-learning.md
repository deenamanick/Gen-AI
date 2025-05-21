Here are some practical exercises for **unsupervised learning**, covering clustering, dimensionality reduction, and anomaly detection. These exercises will help you understand key algorithms and how to evaluate them without labeled data.

---

## **1. Clustering Exercises**
### **A. K-Means Clustering**
**Problem:** Group customers into segments based on purchasing behavior.  
**Dataset:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
**Tasks:**  
1. Load and explore the data (check distributions, missing values).  
2. Select relevant features (e.g., `Annual Income`, `Spending Score`).  
3. **Standardize the data** (K-Means is distance-based).  
4. Apply **K-Means clustering** and find the optimal `k` using:
   - **Elbow Method** (inertia vs. `k`)  
   - **Silhouette Score**  
5. Visualize clusters (scatter plot with colors for clusters).  
6. (Optional) Compare with **Hierarchical Clustering (Dendrogram)**.  

---

### **B. DBSCAN (Density-Based Clustering)**
**Problem:** Detect anomalies or dense regions in a dataset.  
**Dataset:** [Credit Card Transactions](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (use only non-fraud data for clustering).  
**Tasks:**  
1. Preprocess data (scale features, handle outliers).  
2. Apply **DBSCAN** and tune `eps` and `min_samples`.  
3. Identify noise points (potential anomalies).  
4. Compare with **K-Means**—does DBSCAN find more meaningful clusters?  
5. (Optional) Use **t-SNE** for visualization if features are high-dimensional.  

---

## **2. Dimensionality Reduction Exercises**
### **A. PCA (Principal Component Analysis)**
**Problem:** Reduce features while preserving variance in a high-dimensional dataset.  
**Dataset:** [Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)  
**Tasks:**  
1. Standardize the data (PCA is sensitive to scale).  
2. Apply **PCA** and plot **explained variance ratio** (scree plot).  
3. Choose the number of components (e.g., 95% variance).  
4. Visualize data in 2D/3D using the first 2-3 principal components.  
5. (Optional) Apply **K-Means on reduced data**—do clusters improve?  

---

### **B. t-SNE (Non-linear Dimensionality Reduction)**
**Problem:** Visualize high-dimensional data in 2D (e.g., MNIST digits).  
**Dataset:** [MNIST Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)  
**Tasks:**  
1. Load the dataset (images flattened into 64D vectors).  
2. Apply **t-SNE** with `perplexity=30` and visualize in 2D.  
3. Compare with **PCA**—does t-SNE separate classes better?  
4. (Optional) Use **UMAP** (another non-linear method) and compare.  

---

## **3. Anomaly Detection Exercises**
### **A. Isolation Forest**
**Problem:** Detect fraudulent transactions.  
**Dataset:** [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Tasks:**  
1. Train **Isolation Forest** on normal transactions (class `0`).  
2. Predict anomalies (class `1`) and evaluate using:
   - **Confusion Matrix**  
   - **Precision-Recall Curve**  
3. Compare with **DBSCAN** (which works better for fraud?).  

---

### **B. Gaussian Mixture Models (GMM)**
**Problem:** Model customer spending patterns and detect outliers.  
**Dataset:** [Mall Customer Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
**Tasks:**  
1. Fit a **GMM** with `n_components=3`.  
2. Compute **log-likelihood scores** per sample.  
3. Flag samples with very low likelihood as outliers.  
4. (Optional) Compare with **One-Class SVM**.  

---

## **4. Association Rule Learning (Market Basket Analysis)**
### **A. Apriori Algorithm**
**Problem:** Find frequently bought-together items in a supermarket.  
**Dataset:** [Groceries Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)  
**Tasks:**  
1. Preprocess data (convert to transactional format).  
2. Apply **Apriori** to find frequent itemsets (`min_support=0.01`).  
3. Generate **association rules** (`min_confidence=0.3`).  
4. Interpret rules (e.g., "If {milk, bread}, then {eggs}").  

---

## **Tips for Success**
- **Visualize clusters** (PCA/t-SNE helps in 2D/3D plots).  
- **Scale features** (K-Means, PCA, DBSCAN need scaling).  
- **Experiment with hyperparameters** (`k` in K-Means, `eps` in DBSCAN).  
- **Evaluate clustering** (Silhouette Score, Davies-Bouldin Index).  
