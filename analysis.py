"""Compare SymNMF and K-means (HW1) clustering on a dataset.

Usage: python3 analysis.py <k> <file_name>
Prints the silhouette score of each method and the ARI between them.
"""
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

import symnmf

MAX_ITER = 300
EPSILON = 1e-4


def euclidean_distance(p1, p2):
    """Euclidean distance between two points.

    Input: p1, p2 - equal-length sequences of floats.
    Output: the distance (float).
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))))


def closest_centroid(point, centroids):
    """Index of the centroid nearest to point (first one on ties).

    Input: point - sequence of floats, centroids - list of points.
    Output: int index into centroids.
    """
    best, best_dist = 0, float("inf")
    for i, centroid in enumerate(centroids):
        dist = euclidean_distance(point, centroid)
        if dist < best_dist:
            best, best_dist = i, dist
    return best


def kmeans_step(points, centroids):
    """One K-means iteration (HW1): assign each point to its closest
    centroid and recompute every centroid as its cluster mean. An empty
    cluster keeps its previous centroid, as in HW1.

    Input: points - list of points, centroids - list of K points.
    Output: list of K new centroids.
    """
    d, K = len(points[0]), len(centroids)
    sums = [[0.0] * d for _ in range(K)]
    counts = [0] * K
    for point in points:
        idx = closest_centroid(point, centroids)
        counts[idx] += 1
        for j in range(d):
            sums[idx][j] += point[j]
    return [[s / counts[i] for s in sums[i]] if counts[i] else
            centroids[i][:] for i in range(K)]


def kmeans(points, K):
    """K-means from HW1, with the first K points as the initial centroids,
    max_iter = 300 and convergence when every centroid moves less than
    epsilon = 1e-4.

    Input: points - list of points (lists of floats), K - clusters.
    Output: list of cluster labels, one per point.
    """
    centroids = [p[:] for p in points[:K]]
    for _ in range(MAX_ITER):
        new_centroids = kmeans_step(points, centroids)
        converged = all(euclidean_distance(old, new) < EPSILON
                        for old, new in zip(centroids, new_centroids))
        centroids = new_centroids
        if converged:
            break
    return [closest_centroid(p, centroids) for p in points]


def symnmf_labels(X, k):
    """Cluster with SymNMF: each point gets the column of the highest
    association score in its row of H (section 1.5).

    Input: X - N x d numpy array, k - clusters.
    Output: numpy array of N labels.
    """
    H = symnmf.run_symnmf(X, k)
    return np.argmax(H, axis=1)


def safe_silhouette(X, labels):
    """silhouette_score that never raises: it needs 2 <= #labels <= N - 1.
    On failure prints the error and exits.

    Input: X - N x d numpy array, labels - N cluster labels.
    Output: the silhouette score (float).
    """
    n_labels = len(set(labels))
    if not 2 <= n_labels <= len(X) - 1:
        symnmf.fail("An Error Has Occurred: silhouette_score requires "
                    "2 <= number of clusters <= N - 1, got %d" % n_labels)
    try:
        return silhouette_score(X, labels)
    except Exception as err:
        symnmf.fail("An Error Has Occurred: %s" % err)


def main():
    """Parse arguments, run both methods and print the scores."""
    if len(sys.argv) != 3:
        symnmf.fail()
    k = symnmf.parse_k(sys.argv[1])
    X = symnmf.read_points(sys.argv[2])
    symnmf.validate_k(k, X.shape[0])
    try:
        nmf_labels = symnmf_labels(X, k)
    except Exception:
        symnmf.fail()
    km_labels = kmeans(X.tolist(), k)
    nmf_score = safe_silhouette(X, nmf_labels)
    km_score = safe_silhouette(X, km_labels)
    print("nmf: %.4f" % nmf_score)
    print("kmeans: %.4f" % km_score)
    print("ari: %.4f" % adjusted_rand_score(nmf_labels, km_labels))


if __name__ == "__main__":
    main()
