from sklearn.cluster import KMeans

def determine_optimal_clusters(X, max_clusters=10):
    """Determine the optimal number of clusters using the Elbow method."""
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

def apply_kmeans(X, n_clusters=3):
    """Apply KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters
