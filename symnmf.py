"""SymNMF clustering: Python interface to the symnmfmodule C extension.

Usage: python3 symnmf.py <k> <goal> <file_name>
goal is one of: symnmf, sym, ddg, norm.
"""
import sys

import numpy as np

import symnmfmodule

ERROR_MSG = "An Error Has Occurred"
K_ERROR_MSG = "Incorrect number of clusters!"
GOALS = ("symnmf", "sym", "ddg", "norm")
MAX_ITER = 300
EPSILON = 1e-4


def fail(msg=ERROR_MSG):
    """Print msg and exit with status 1.

    Input: msg - message to print (default: the general error message).
    """
    print(msg)
    sys.exit(1)


def parse_k(text):
    """Parse k as a natural number greater than 1 (the k < N check is done
    once the data is read).

    Input: text - the command line string.
    Output: k as int. Exits with the clusters error message if invalid.
    """
    # Accept a natural number written as a whole decimal: 3, 3., 3.0, 3.000.
    # Checked on the string itself, since float() would round values such
    # as 3.000000000000000000000000000000000000001 to 3.
    whole, _, frac = text.partition(".")
    if not (whole.isascii() and whole.isdigit()) or frac.strip("0"):
        fail(K_ERROR_MSG)
    k = int(whole)
    if k <= 1:
        fail(K_ERROR_MSG)
    return k


def read_points(file_name):
    """Read comma-separated data points, one point per line.

    Input: file_name - path to the input file.
    Output: N x d numpy array of floats (d may be 1).
            Exits with the general error message on failure.
    """
    try:
        return np.loadtxt(file_name, delimiter=",", ndmin=2, dtype=float)
    except Exception:
        fail()


def validate_k(k, n):
    """Exit with the clusters error message unless 1 < k < n.

    Input: k - number of clusters, n - number of data points.
    """
    if not 1 < k < n:
        fail(K_ERROR_MSG)


def init_h(W, k):
    """Randomly initialize H (section 1.4.1) with values uniformly drawn
    from [0, 2 * sqrt(m / k)], m being the mean of all entries of W.

    Input: W - n x n normalized similarity (numpy array), k - clusters.
    Output: n x k numpy array.
    """
    m = np.mean(W)
    return np.random.uniform(0, 2 * np.sqrt(m / k), size=(W.shape[0], k))


def run_symnmf(X, k):
    """Run the full SymNMF algorithm: W = norm(X), random initial H and
    the C optimization of H.

    Input: X - N x d numpy array, k - number of clusters.
    Output: final H as an N x k numpy array.
    """
    np.random.seed(1234)
    W = np.array(symnmfmodule.norm(X.tolist()))
    H = init_h(W, k)
    return np.array(symnmfmodule.symnmf(H.tolist(), W.tolist(),
                                        MAX_ITER, EPSILON))


def compute_goal(goal, X, k):
    """Compute the matrix requested by goal.

    Input: goal - one of GOALS, X - N x d numpy array, k - clusters.
    Output: the resulting matrix (list of lists or numpy array).
    """
    if goal == "symnmf":
        return run_symnmf(X, k)
    funcs = {"sym": symnmfmodule.sym, "ddg": symnmfmodule.ddg,
             "norm": symnmfmodule.norm}
    return funcs[goal](X.tolist())


def print_matrix(mat):
    """Print a matrix, comma separated, one row per line, 4 decimals.

    Input: mat - 2D iterable of numbers.
    """
    for row in mat:
        print(",".join("%.4f" % val for val in row))


def main():
    """Parse arguments, compute the requested goal and print the result.
    Any error prints a message and exits with status 1.
    """
    if len(sys.argv) != 4:
        fail()
    k = parse_k(sys.argv[1])
    goal = sys.argv[2]
    if goal not in GOALS:
        fail()
    X = read_points(sys.argv[3])
    validate_k(k, X.shape[0])
    try:
        result = compute_goal(goal, X, k)
    except Exception:
        fail()
    print_matrix(result)


if __name__ == "__main__":
    main()
