#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>

/*
 * alloc_matrix - allocate a zero-initialized rows x cols matrix of doubles.
 * The data is one contiguous block; row pointers point into it.
 * Input: rows, cols - dimensions (positive).
 * Output: the matrix, or NULL on allocation failure. Free with free_matrix.
 */
double **alloc_matrix(int rows, int cols);

/*
 * free_matrix - free a matrix created by alloc_matrix. NULL is ignored.
 * Input: mat - the matrix.
 */
void free_matrix(double **mat);

/*
 * sq_dist - squared Euclidean distance between two d-dimensional points.
 * Input: a, b - the points, d - dimension. Output: sum of (a_i - b_i)^2.
 */
double sq_dist(const double *a, const double *b, int d);

/*
 * sym - compute the similarity matrix A (section 1.1).
 * Input: X - n x d data points, n - number of points, d - dimension.
 * Output: new n x n matrix A, or NULL on allocation failure.
 */
double **sym(double **X, int n, int d);

/*
 * ddg - compute the diagonal degree matrix D (section 1.2).
 * Input: X - n x d data points, n - number of points, d - dimension.
 * Output: new n x n diagonal matrix D, or NULL on allocation failure.
 */
double **ddg(double **X, int n, int d);

/*
 * norm - compute the normalized similarity matrix W = D^-1/2 A D^-1/2
 * (section 1.3).
 * Input: X - n x d data points, n - number of points, d - dimension.
 * Output: new n x n matrix W, or NULL on allocation failure.
 */
double **norm(double **X, int n, int d);

/*
 * mat_mult - compute C = A * B into an existing matrix.
 * Input: A - r x m, B - m x c, C - r x c output (overwritten), r, m, c.
 */
void mat_mult(double **A, double **B, double **C, int r, int m, int c);

/*
 * gram - compute G = H^T H into an existing matrix.
 * Input: H - n x k, G - k x k output (overwritten), n, k - dimensions.
 */
void gram(double **H, double **G, int n, int k);

/*
 * update_step - one multiplicative update of H (section 1.4.2).
 * Input: H - n x k current matrix, W - n x n, next - n x k output,
 *        bufs - scratch matrices {WH (n x k), H^T H (k x k), H H^T H
 *        (n x k)}, n, k - dimensions.
 * Output: ||next - H||_F^2.
 */
double update_step(double **H, double **W, double **next,
                   double **bufs[3], int n, int k);

/*
 * symnmf - optimize H by the multiplicative update rule (section 1.4)
 * until max_iter iterations or ||H(t+1) - H(t)||_F^2 < eps.
 * Input: H - n x k initial matrix, W - n x n normalized similarity,
 *        n, k - dimensions, max_iter, eps - convergence parameters.
 * Output: new n x k final matrix H, or NULL on allocation failure.
 *         The input H is not modified.
 */
double **symnmf(double **H, double **W, int n, int k, int max_iter,
                double eps);

/*
 * read_values - read all comma-separated doubles of an open file.
 * Input: fp - open file, count - output number of values,
 *        d - output number of values in the first line (the dimension).
 * Output: malloc'ed array of *count values, or NULL on failure.
 */
double *read_values(FILE *fp, int *count, int *d);

/*
 * read_points - read a data file (one comma-separated point per line).
 * Input: path - file path, n, d - outputs for the matrix dimensions.
 * Output: new n x d matrix, or NULL on any failure.
 */
double **read_points(const char *path, int *n, int *d);

/*
 * print_matrix - print a matrix comma separated, one row per line, '%.4f'.
 * Input: mat - rows x cols matrix, rows, cols - dimensions.
 */
void print_matrix(double **mat, int rows, int cols);

#endif
