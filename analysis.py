import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

import symnmf

KMEANS_MAX_ITER = 300
KMEANS_EPSILON = 1e-4


def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points.

    Input: p1, p2 - two points with the same number of coordinates.
    Output: the distance between them.
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))))


def closest_centroid(point, centroids):
    """Finds the centroid that is nearest to a point. On a tie the first
    one wins.

    Input: point - a single point, centroids - the current centroids.
    Output: the index of the nearest centroid.
    """
    best, best_dist = 0, float("inf")
    for i, centroid in enumerate(centroids):
        dist = euclidean_distance(point, centroid)
        if dist < best_dist:
            best, best_dist = i, dist
    return best


def kmeans_step(points, centroids):
    """Runs one K-means iteration: every point joins its nearest centroid
    and every centroid moves to the average of its points. A cluster that
    ends up empty keeps its old centroid, like in HW1.

    Input: points - all data points, centroids - the current K centroids.
    Output: the list of K new centroids.
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
    """K-means from HW1. Starts from the first K points and stops when
    every centroid moves less than epsilon, or after max_iter iterations.

    Input: points - the data points as lists, K - number of clusters.
    Output: the cluster label of every point.
    """
    centroids = [p[:] for p in points[:K]]
    for _ in range(KMEANS_MAX_ITER):
        new_centroids = kmeans_step(points, centroids)
        converged = all(euclidean_distance(old, new) < KMEANS_EPSILON
                        for old, new in zip(centroids, new_centroids))
        centroids = new_centroids
        if converged:
            break
    return [closest_centroid(p, centroids) for p in points]


def symnmf_labels(X, k):
    """Clusters the points with SymNMF: each point goes to the cluster
    (column of H) where it has the highest score.

    Input: X - the data points, k - number of clusters.
    Output: a numpy array with the cluster label of every point.
    """
    H = symnmf.run_symnmf(X, k)
    return np.argmax(H, axis=1)


def safe_silhouette(X, labels):
    """Calculates the silhouette score without letting it crash.
    silhouette_score only works when there are between 2 and N - 1
    different labels, so this is checked first. If anything still goes
    wrong, the error is printed and the program exits.

    Input: X - the data points, labels - the cluster label of every point.
    Output: the silhouette score.
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
