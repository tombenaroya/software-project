#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define ERROR_MSG "An Error Has Occurred\n"
#define BETA 0.5
/* Added to the update denominator to avoid division by zero (forum). */
#define DENOM_EPS 1e-12

/*
 * alloc_matrix - allocate a zero-initialized rows x cols matrix of doubles.
 * One contiguous data block is used, so the matrix is freed with two frees.
 * Input: rows, cols - dimensions. Output: the matrix or NULL on failure.
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
 * free_matrix - free a matrix created by alloc_matrix. NULL is ignored.
 * Input: mat - the matrix.
 */
void free_matrix(double **mat)
{
    if (mat == NULL)
        return;
    free(mat[0]);
    free(mat);
}

/*
 * sq_dist - squared Euclidean distance between two d-dimensional points.
 * Input: a, b - the points, d - dimension. Output: sum of (a_i - b_i)^2.
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
 * sym - compute the similarity matrix: a_ij = exp(-||xi - xj||^2 / 2) for
 * i != j and a_ii = 0.
 * Input: X - n x d points, n, d - dimensions.
 * Output: new n x n matrix A, or NULL on allocation failure.
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
 * ddg - compute the diagonal degree matrix: d_ii = sum_j a_ij, zero
 * elsewhere.
 * Input: X - n x d points, n, d - dimensions.
 * Output: new n x n matrix D, or NULL on allocation failure.
 */
double **ddg(double **X, int n, int d)
{
    double **A = sym(X, n, d), **D;
    int i, j;
    if (A == NULL)
        return NULL;
    D = alloc_matrix(n, n);
    if (D != NULL) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                D[i][i] += A[i][j];
    }
    free_matrix(A);
    return D;
}

/*
 * norm - compute W = D^-1/2 A D^-1/2. Since D is diagonal,
 * w_ij = a_ij / sqrt(d_i * d_j); a zero degree contributes a zero factor.
 * Input: X - n x d points, n, d - dimensions.
 * Output: new n x n matrix W, or NULL on allocation failure.
 */
double **norm(double **X, int n, int d)
{
    double **A = sym(X, n, d), **W = NULL, *inv_sqrt = NULL;
    int i, j;
    if (A != NULL) {
        W = alloc_matrix(n, n);
        inv_sqrt = (double *)calloc(n, sizeof(double));
    }
    if (W != NULL && inv_sqrt != NULL) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                inv_sqrt[i] += A[i][j];
            inv_sqrt[i] = inv_sqrt[i] > 0 ? 1.0 / sqrt(inv_sqrt[i]) : 0.0;
        }
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                W[i][j] = inv_sqrt[i] * A[i][j] * inv_sqrt[j];
    } else {
        free_matrix(W);
        W = NULL;
    }
    free(inv_sqrt);
    free_matrix(A);
    return W;
}

/*
 * mat_mult - compute C = A * B into an existing matrix.
 * Input: A - r x m, B - m x c, C - r x c output (overwritten), r, m, c.
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
 * gram - compute G = H^T H into an existing matrix.
 * Input: H - n x k, G - k x k output (overwritten), n, k - dimensions.
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
 * update_step - one multiplicative update of section 1.4.2:
 * next_ij = H_ij * (1 - beta + beta * (WH)_ij / (H H^T H)_ij).
 * H H^T H is computed as H (H^T H), which costs O(n k^2) instead of O(n^2 k).
 * Input: H - n x k, W - n x n, next - n x k output, bufs - scratch
 *        {WH (n x k), G = H^T H (k x k), HG (n x k)}, n, k.
 * Output: ||next - H||_F^2.
 */
double update_step(double **H, double **W, double **next,
                          double **bufs[3], int n, int k)
{
    double **WH = bufs[0], **G = bufs[1], **HG = bufs[2];
    double diff, dist = 0.0;
    int i, j;
    mat_mult(W, H, WH, n, n, k);
    gram(H, G, n, k);
    mat_mult(H, G, HG, n, k, k);
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            next[i][j] = H[i][j] * (1.0 - BETA + BETA * WH[i][j]
                                    / (HG[i][j] + DENOM_EPS));
            diff = next[i][j] - H[i][j];
            dist += diff * diff;
        }
    }
    return dist;
}

/*
 * symnmf - iterate update_step until max_iter or convergence (< eps).
 * Input: H - n x k initial matrix (not modified), W - n x n, n, k,
 *        max_iter, eps. Output: new n x k matrix, or NULL on failure.
 */
double **symnmf(double **H, double **W, int n, int k, int max_iter,
                double eps)
{
    double **cur = alloc_matrix(n, k), **next = alloc_matrix(n, k), **tmp;
    double **bufs[3];
    int it, ok;
    bufs[0] = alloc_matrix(n, k);
    bufs[1] = alloc_matrix(k, k);
    bufs[2] = alloc_matrix(n, k);
    ok = cur && next && bufs[0] && bufs[1] && bufs[2];
    if (ok) {
        memcpy(cur[0], H[0], (size_t)n * k * sizeof(double));
        for (it = 0; it < max_iter; it++) {
            double dist = update_step(cur, W, next, bufs, n, k);
            tmp = cur; /* the newest H always ends up in cur */
            cur = next;
            next = tmp;
            if (dist < eps)
                break;
        }
    }
    free_matrix(next);
    free_matrix(bufs[0]);
    free_matrix(bufs[1]);
    free_matrix(bufs[2]);
    if (!ok) {
        free_matrix(cur);
        return NULL;
    }
    return cur;
}

/*
 * read_values - read all comma-separated doubles of a file into a growing
 * array, counting the values in the first line as the dimension.
 * Input: fp - open file, count, d - outputs.
 * Output: malloc'ed array of *count values, or NULL on failure/empty file.
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
 * read_points - read a data file into an n x d matrix.
 * Input: path - file path, n, d - outputs.
 * Output: new matrix, or NULL on any failure.
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
 * print_matrix - print a matrix, comma separated, each row on its own line,
 * with 4 decimal places.
 * Input: mat - rows x cols matrix, rows, cols.
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
 * main - usage: ./symnmf <goal> <file_name>, goal in {sym, ddg, norm}.
 * Prints the requested matrix. On any error prints "An Error Has Occurred"
 * and returns 1.
 */
int main(int argc, char *argv[])
{
    double **X, **res = NULL;
    int n, d;
    if (argc != 3) {
        printf("%s", ERROR_MSG);
        return 1;
    }
    if (strcmp(argv[1], "sym") && strcmp(argv[1], "ddg")
        && strcmp(argv[1], "norm")) {
        printf("%s", ERROR_MSG);
        return 1;
    }
    X = read_points(argv[2], &n, &d);
    if (X != NULL) {
        if (strcmp(argv[1], "sym") == 0)
            res = sym(X, n, d);
        else if (strcmp(argv[1], "ddg") == 0)
            res = ddg(X, n, d);
        else
            res = norm(X, n, d);
    }
    free_matrix(X);
    if (res == NULL) {
        printf("%s", ERROR_MSG);
        return 1;
    }
    print_matrix(res, n, n);
    free_matrix(res);
    return 0;
}
