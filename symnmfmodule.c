#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/*
 * fill_row - copy a Python list of floats into a C row.
 * Input: row - the Python object, dst - C row of length cols, cols.
 * Output: 0 on success, -1 with a Python exception set.
 */
static int fill_row(PyObject *row, double *dst, int cols)
{
    Py_ssize_t j;
    if (!PyList_Check(row) || PyList_Size(row) != cols) {
        PyErr_SetString(PyExc_ValueError, "expected equal-length rows");
        return -1;
    }
    for (j = 0; j < cols; j++)
        dst[j] = PyFloat_AsDouble(PyList_GetItem(row, j));
    return PyErr_Occurred() ? -1 : 0;
}

/*
 * list_to_matrix - convert a non-empty Python list of equal-length,
 * non-empty lists of floats into a new C matrix.
 * Input: obj - the Python object, rows, cols - outputs for dimensions.
 * Output: new matrix, or NULL with a Python exception set.
 */
static double **list_to_matrix(PyObject *obj, int *rows, int *cols)
{
    double **mat;
    Py_ssize_t i;
    if (!PyList_Check(obj) || PyList_Size(obj) < 1
        || !PyList_Check(PyList_GetItem(obj, 0))
        || PyList_Size(PyList_GetItem(obj, 0)) < 1) {
        PyErr_SetString(PyExc_ValueError, "expected a non-empty 2D list");
        return NULL;
    }
    *rows = (int)PyList_Size(obj);
    *cols = (int)PyList_Size(PyList_GetItem(obj, 0));
    mat = alloc_matrix(*rows, *cols);
    if (mat == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (i = 0; i < *rows; i++) {
        if (fill_row(PyList_GetItem(obj, i), mat[i], *cols) < 0) {
            free_matrix(mat);
            return NULL;
        }
    }
    return mat;
}

/*
 * matrix_to_list - convert a C matrix into a new Python list of lists.
 * Input: mat - rows x cols matrix, rows, cols.
 * Output: new reference to the list, or NULL with a Python exception set.
 */
static PyObject *matrix_to_list(double **mat, int rows, int cols)
{
    PyObject *list = PyList_New(rows), *row, *val;
    int i, j;
    if (list == NULL)
        return NULL;
    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        if (row == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, row); /* steals the reference to row */
        for (j = 0; j < cols; j++) {
            val = PyFloat_FromDouble(mat[i][j]);
            if (val == NULL) {
                Py_DECREF(list);
                return NULL;
            }
            PyList_SET_ITEM(row, j, val);
        }
    }
    return list;
}

/*
 * run_on_points - shared body of sym/ddg/norm: parse the data points,
 * apply func and return the resulting n x n matrix as a Python list.
 * Input: args - Python args tuple (X), func - one of sym, ddg, norm.
 * Output: new Python list, or NULL with an exception set.
 */
static PyObject *run_on_points(PyObject *args,
                               double **(*func)(double **, int, int))
{
    PyObject *obj, *result;
    double **X, **res;
    int n, d;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    X = list_to_matrix(obj, &n, &d);
    if (X == NULL)
        return NULL;
    res = func(X, n, d);
    free_matrix(X);
    if (res == NULL)
        return PyErr_NoMemory();
    result = matrix_to_list(res, n, n);
    free_matrix(res);
    return result;
}

/* sym(X) -> similarity matrix A as a list of lists. */
static PyObject *py_sym(PyObject *self, PyObject *args)
{
    (void)self;
    return run_on_points(args, sym);
}

/* ddg(X) -> diagonal degree matrix D as a list of lists. */
static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    (void)self;
    return run_on_points(args, ddg);
}

/* norm(X) -> normalized similarity matrix W as a list of lists. */
static PyObject *py_norm(PyObject *self, PyObject *args)
{
    (void)self;
    return run_on_points(args, norm);
}

/*
 * py_symnmf - symnmf(H, W, max_iter, eps) -> final H as a list of lists.
 * Input: H - n x k initial matrix, W - n x n normalized similarity,
 *        max_iter, eps - convergence parameters.
 * Output: new Python list, or NULL with an exception set.
 */
static PyObject *py_symnmf(PyObject *self, PyObject *args)
{
    PyObject *h_obj, *w_obj, *result = NULL;
    double **H, **W = NULL, **res = NULL, eps;
    int n, k, wn, wm, max_iter;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOid", &h_obj, &w_obj, &max_iter, &eps))
        return NULL;
    H = list_to_matrix(h_obj, &n, &k);
    if (H != NULL)
        W = list_to_matrix(w_obj, &wn, &wm);
    if (W != NULL && (wn != n || wm != n))
        PyErr_SetString(PyExc_ValueError, "dimension mismatch");
    else if (W != NULL) {
        res = symnmf(H, W, n, k, max_iter, eps);
        if (res == NULL)
            PyErr_NoMemory();
        else
            result = matrix_to_list(res, n, k);
    }
    free_matrix(H);
    free_matrix(W);
    free_matrix(res);
    return result;
}

static PyMethodDef symnmf_methods[] = {
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS,
     "symnmf(H, W, max_iter, eps) -> optimized H (list of lists)"},
    {"sym", (PyCFunction)py_sym, METH_VARARGS,
     "sym(X) -> similarity matrix (list of lists)"},
    {"ddg", (PyCFunction)py_ddg, METH_VARARGS,
     "ddg(X) -> diagonal degree matrix (list of lists)"},
    {"norm", (PyCFunction)py_norm, METH_VARARGS,
     "norm(X) -> normalized similarity matrix (list of lists)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "SymNMF clustering C extension",
    -1,
    symnmf_methods,
    NULL, NULL, NULL, NULL
};

/* PyInit_symnmfmodule - module entry point called on import. */
PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmf_module);
}
