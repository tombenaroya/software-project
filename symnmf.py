import sys

import numpy as np

import symnmfmodule

ERROR_MSG = "An Error Has Occurred"
K_ERROR_MSG = "Incorrect number of clusters!"
# Goals computed directly by the C module from the data points.
MATRIX_GOALS = {"sym": symnmfmodule.sym, "ddg": symnmfmodule.ddg,
                "norm": symnmfmodule.norm}
GOALS = ("symnmf",) + tuple(MATRIX_GOALS)
SYMNMF_MAX_ITER = 300
SYMNMF_EPSILON = 1e-4


def fail(msg=ERROR_MSG):
    """Prints an error message and exits the program with status 1.

    Input: msg - the message to print (the general error by default).
    """
    print(msg)
    sys.exit(1)


def parse_k(text):
    """Reads k from the command line and makes sure it is a natural
    number bigger than 1. Comparing k to N happens later, once the file
    has been read.

    Input: text - k as it was typed on the command line.
    Output: k as an int. Exits with an error if k is not valid.
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
    """Reads the data points from the input file, one point per line.

    Input: file_name - path to the input file.
    Output: an N x d numpy array (d can be 1). Exits with an error if
            the file cannot be read.
    """
    try:
        return np.loadtxt(file_name, delimiter=",", ndmin=2, dtype=float)
    except Exception:
        fail()


def validate_k(k, n):
    """Makes sure that 1 < k < N, otherwise exits with an error.

    Input: k - number of clusters, n - number of data points.
    """
    if not 1 < k < n:
        fail(K_ERROR_MSG)


def init_h(W, k):
    """Creates the random starting H, with values taken uniformly from
    [0, 2 * sqrt(m / k)], where m is the average of all entries in W.

    Input: W - the normalized similarity matrix, k - number of clusters.
    Output: an n x k numpy array.
    """
    m = np.mean(W)
    return np.random.uniform(0, 2 * np.sqrt(m / k), size=(W.shape[0], k))


def run_symnmf(X, k):
    """Runs the whole SymNMF algorithm: builds W with the C norm(),
    creates a random starting H and lets the C code optimize it.

    Input: X - the data points (N x d numpy array), k - number of clusters.
    Output: the final H as an N x k numpy array.
    """
    np.random.seed(1234)
    W = np.array(symnmfmodule.norm(X.tolist()))
    H = init_h(W, k)
    return np.array(symnmfmodule.symnmf(H.tolist(), W.tolist(),
                                        SYMNMF_MAX_ITER, SYMNMF_EPSILON))


def compute_goal(goal, X, k):
    """Calculates the matrix that the user asked for.

    Input: goal - symnmf, sym, ddg or norm, X - the data points,
           k - number of clusters.
    Output: the requested matrix.
    """
    if goal == "symnmf":
        return run_symnmf(X, k)
    return MATRIX_GOALS[goal](X.tolist())


def print_matrix(mat):
    """Prints a matrix with each row on its own line, values separated
    by commas and shown with 4 decimal places.

    Input: mat - the matrix to print.
    """
    for row in mat:
        print(",".join("%.4f" % val for val in row))


def main():
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
