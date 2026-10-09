#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define ERROR_MSG "An Error Has Occurred\n"
#define BETA 0.5
#define DENOM_EPS 1e-12

/*
 * Creates a rows x cols matrix of doubles, filled with zeros.
 * All the numbers live in one block of memory and each row points into it,
 * so freeing the matrix only takes two calls (see free_matrix).
 * Input:  rows, cols - the size of the matrix.
 * Output: the new matrix, or NULL if memory allocation failed.
 */
double **alloc_matrix(int rows, int cols)
{
    double **mat;
    double *data;
    int i;
    mat = (double **)malloc(rows * sizeof(double *));
    data = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (mat == NULL || data == NULL) {
        free(mat);
        free(data);
        return NULL;
    }
    for (i = 0; i < rows; i++)
        mat[i] = data + (size_t)i * cols;
    return mat;
}

/*
 * Releases a matrix that was created with alloc_matrix.
 * Passing NULL is safe and does nothing.
 * Input:  mat - the matrix to free.
 */
void free_matrix(double **mat)
{
    if (mat == NULL)
        return;
    free(mat[0]);
    free(mat);
}

/*
 * Calculates the squared Euclidean distance between two points.
 * Input:  a, b - the two points, d - how many coordinates each point has.
 * Output: the sum of (a[i] - b[i])^2 over all coordinates.
 */
double sq_dist(const double *a, const double *b, int d)
{
    double sum = 0.0, diff;
    int i;
    for (i = 0; i < d; i++) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/*
 * Builds the similarity matrix A of the data points:
 * a_ij = exp(-||xi - xj||^2 / 2) when i != j, and 0 on the diagonal.
 * Input:  X - the n data points, each with d coordinates.
 * Output: a new n x n matrix A, or NULL if memory allocation failed.
 */
double **sym(double **X, int n, int d)
{
    double **A = alloc_matrix(n, n);
    int i, j;
    if (A == NULL)
        return NULL;
    /* A is symmetric, so compute the upper triangle and mirror it. */
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            A[i][j] = exp(-sq_dist(X[i], X[j], d) / 2.0);
            A[j][i] = A[i][j];
        }
    }
    return A;
}

/*
 * Calculates the degree of every point, which is the sum of its row in A.
 * Input:  A - the n x n similarity matrix, n - the number of points.
 * Output: a new array of n degrees, or NULL if memory allocation failed.
 */
double *degrees(double **A, int n)
{
    double *deg = (double *)calloc(n, sizeof(double));
    int i, j;
    if (deg == NULL)
        return NULL;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            deg[i] += A[i][j];
    return deg;
}

/*
 * Builds the diagonal degree matrix D: the degree of each point on the
 * diagonal, and zeros everywhere else.
 * Input:  X - the n data points, each with d coordinates.
 * Output: a new n x n matrix D, or NULL if memory allocation failed.
 */
double **ddg(double **X, int n, int d)
{
    double **A = sym(X, n, d), **D = NULL, *deg = NULL;
    int i;
    if (A != NULL)
        deg = degrees(A, n);
    if (deg != NULL)
        D = alloc_matrix(n, n);
    if (D != NULL)
        for (i = 0; i < n; i++)
            D[i][i] = deg[i];
    free(deg);
    free_matrix(A);
    return D;
}

/*
 * Builds the normalized similarity matrix W = D^-1/2 * A * D^-1/2.
 * Because D is diagonal there is no need for real matrix products:
 * w_ij is simply a_ij / sqrt(d_i * d_j).
 * Input:  X - the n data points, each with d coordinates.
 * Output: a new n x n matrix W, or NULL if memory allocation failed.
 */
double **norm(double **X, int n, int d)
{
    double **A = sym(X, n, d), **W = NULL, *inv_sqrt = NULL;
    int i, j;
    if (A != NULL)
        inv_sqrt = degrees(A, n);
    if (inv_sqrt != NULL)
        W = alloc_matrix(n, n);
    if (W != NULL) {
        /* turn each degree d_i into d_i^-1/2 in place */
        for (i = 0; i < n; i++)
            inv_sqrt[i] = inv_sqrt[i] > 0 ? 1.0 / sqrt(inv_sqrt[i]) : 0.0;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                W[i][j] = inv_sqrt[i] * A[i][j] * inv_sqrt[j];
    }
    free(inv_sqrt);
    free_matrix(A);
    return W;
}

/*
 * Multiplies two matrices, C = A * B, and writes the result into C.
 * Input:  A - an r x m matrix, B - an m x c matrix,
 *         C - an existing r x c matrix that receives the result.
 */
void mat_mult(double **A, double **B, double **C, int r, int m, int c)
{
    int i, j, l;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            C[i][j] = 0.0;
            for (l = 0; l < m; l++)
                C[i][j] += A[i][l] * B[l][j];
        }
    }
}

/*
 * Calculates H^T * H and writes the result into G.
 * Input:  H - an n x k matrix, G - an existing k x k matrix for the result.
 */
void gram(double **H, double **G, int n, int k)
{
    int i, j, l;
    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            G[i][j] = 0.0;
            for (l = 0; l < n; l++)
                G[i][j] += H[l][i] * H[l][j];
        }
    }
}

/*
 * Allocates the helper matrices that update_step needs on every
 * iteration, so they are created once and not on every step.
 * Input:  buf - the struct to fill, n, k - the size of H.
 * Output: 1 on success, 0 if memory allocation failed (in that case
 *         nothing is left allocated).
 */
int alloc_buffers(UpdateBuffers *buf, int n, int k)
{
    buf->WH = alloc_matrix(n, k);
    buf->G = alloc_matrix(k, k);
    buf->HG = alloc_matrix(n, k);
    if (buf->WH && buf->G && buf->HG)
        return 1;
    free_buffers(buf);
    return 0;
}

/*
 * Releases the helper matrices created by alloc_buffers.
 * Input:  buf - the struct to clean up.
 */
void free_buffers(UpdateBuffers *buf)
{
    free_matrix(buf->WH);
    free_matrix(buf->G);
    free_matrix(buf->HG);
    buf->WH = buf->G = buf->HG = NULL;
}

/*
 * Performs one update of H:
 * next_ij = H_ij * (1 - beta + beta * (W*H)_ij / (H*H^T*H)_ij).
 * H*H^T*H is calculated as H * (H^T*H), which is much cheaper because
 * H^T*H is only k x k. A tiny epsilon is added to the denominator to
 * avoid dividing by zero.
 * Input:  H - the current n x k matrix, W - the normalized similarity,
 *         next - where the new H is written, buf - helper matrices.
 * Output: the squared Frobenius norm of (next - H), used to check convergence.
 */
double update_step(double **H, double **W, double **next,
                   UpdateBuffers *buf, int n, int k)
{
    double diff, dist = 0.0;
    int i, j;
    mat_mult(W, H, buf->WH, n, n, k);
    gram(H, buf->G, n, k);
    mat_mult(H, buf->G, buf->HG, n, k, k);
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            next[i][j] = H[i][j] * (1.0 - BETA + BETA * buf->WH[i][j]
                                    / (buf->HG[i][j] + DENOM_EPS));
            diff = next[i][j] - H[i][j];
            dist += diff * diff;
        }
    }
    return dist;
}

/*
 * Optimizes H by repeating update_step until H stops changing
 * (the change is smaller than eps) or until max_iter updates were made.
 * Input:  H - the initial n x k matrix (it is not changed),
 *         W - the n x n normalized similarity matrix,
 *         max_iter, eps - when to stop.
 * Output: a new n x k matrix with the final H, or NULL if memory
 *         allocation failed.
 */
double **symnmf(double **H, double **W, int n, int k, int max_iter,
                double eps)
{
    double **cur = alloc_matrix(n, k), **next = alloc_matrix(n, k), **tmp;
    UpdateBuffers buf;
    int it, ok = cur && next && alloc_buffers(&buf, n, k);
    if (ok) {
        memcpy(cur[0], H[0], (size_t)n * k * sizeof(double));
        for (it = 0; it < max_iter; it++) {
            double dist = update_step(cur, W, next, &buf, n, k);
            tmp = cur; /* the newest H always ends up in cur */
            cur = next;
            next = tmp;
            if (dist < eps)
                break;
        }
        free_buffers(&buf);
    }
    free_matrix(next);
    if (!ok) {
        free_matrix(cur);
        return NULL;
    }
    return cur;
}

/*
 * Reads every number from an open data file into one growing array.
 * The number of values on the first line tells us the dimension d.
 * Input:  fp - the open file, count and d - filled with the number of
 *         values read and the dimension.
 * Output: a new array with all the values, or NULL on failure.
 */
double *read_values(FILE *fp, int *count, int *d)
{
    double *vals = NULL, *grown, val;
    int cap = 0, c, first_line = 1;
    *count = 0;
    *d = 0;
    while (fscanf(fp, "%lf", &val) == 1) {
        if (*count == cap) {
            cap = cap ? cap * 2 : 1024;
            grown = (double *)realloc(vals, cap * sizeof(double));
            if (grown == NULL) {
                free(vals);
                return NULL;
            }
            vals = grown;
        }
        vals[(*count)++] = val;
        if (first_line)
            (*d)++;
        c = fgetc(fp);
        if (c == '\n' || c == '\r' || c == EOF)
            first_line = 0;
    }
    return vals;
}

/*
 * Reads a data file (one comma-separated point per line) into a matrix.
 * Input:  path - the file to read, n and d - filled with the number of
 *         points and their dimension.
 * Output: a new n x d matrix, or NULL if anything went wrong.
 */
double **read_points(const char *path, int *n, int *d)
{
    FILE *fp = fopen(path, "r");
    double *vals, **X = NULL;
    int count;
    if (fp == NULL)
        return NULL;
    vals = read_values(fp, &count, d);
    fclose(fp);
    if (vals != NULL && *d > 0 && count % *d == 0) {
        *n = count / *d;
        X = alloc_matrix(*n, *d);
        if (X != NULL)
            memcpy(X[0], vals, (size_t)count * sizeof(double));
    }
    free(vals);
    return X;
}

/*
 * Prints a matrix with each row on its own line, values separated by
 * commas and shown with 4 decimal places.
 * Input:  mat - the matrix, rows, cols - its size.
 */
void print_matrix(double **mat, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            printf(j < cols - 1 ? "%.4f," : "%.4f\n", mat[i][j]);
    }
}

/*
 * Finds the function that matches the goal given on the command line.
 * Input:  goal - "sym", "ddg" or "norm".
 * Output: the matching function, or NULL if the goal is not valid.
 */
GoalFunc goal_function(const char *goal)
{
    if (strcmp(goal, "sym") == 0)
        return sym;
    if (strcmp(goal, "ddg") == 0)
        return ddg;
    if (strcmp(goal, "norm") == 0)
        return norm;
    return NULL;
}

int main(int argc, char *argv[])
{
    GoalFunc func = argc == 3 ? goal_function(argv[1]) : NULL;
    double **X = NULL, **res = NULL;
    int n, d;
    if (func != NULL)
        X = read_points(argv[2], &n, &d);
    if (X != NULL)
        res = func(X, n, d);
    free_matrix(X);
    if (res == NULL) {
        printf("%s", ERROR_MSG);
        return 1;
    }
    print_matrix(res, n, n);
    free_matrix(res);
    return 0;
}
