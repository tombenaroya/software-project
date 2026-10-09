#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>

typedef struct {
    double **WH;
    double **G;
    double **HG;
} UpdateBuffers;

/* A function computing a goal matrix (sym, ddg or norm) from points. */
typedef double **(*GoalFunc)(double **X, int n, int d);

/*
 * Creates a rows x cols matrix of doubles, filled with zeros.
 * All the numbers live in one block of memory and each row points into it,
 * so freeing the matrix only takes two calls (see free_matrix).
 * Input:  rows, cols - the size of the matrix.
 * Output: the new matrix, or NULL if memory allocation failed.
 */
double **alloc_matrix(int rows, int cols);

/*
 * Releases a matrix that was created with alloc_matrix.
 * Passing NULL is safe and does nothing.
 * Input:  mat - the matrix to free.
 */
void free_matrix(double **mat);

/*
 * Calculates the squared Euclidean distance between two points.
 * Input:  a, b - the two points, d - how many coordinates each point has.
 * Output: the sum of (a[i] - b[i])^2 over all coordinates.
 */
double sq_dist(const double *a, const double *b, int d);

/*
 * Builds the similarity matrix A of the data points:
 * a_ij = exp(-||xi - xj||^2 / 2) when i != j, and 0 on the diagonal.
 * Input:  X - the n data points, each with d coordinates.
 * Output: a new n x n matrix A, or NULL if memory allocation failed.
 */
double **sym(double **X, int n, int d);

/*
 * Calculates the degree of every point, which is the sum of its row in A.
 * Input:  A - the n x n similarity matrix, n - the number of points.
 * Output: a new array of n degrees, or NULL if memory allocation failed.
 */
double *degrees(double **A, int n);

/*
 * Builds the diagonal degree matrix D: the degree of each point on the
 * diagonal, and zeros everywhere else.
 * Input:  X - the n data points, each with d coordinates.
 * Output: a new n x n matrix D, or NULL if memory allocation failed.
 */
double **ddg(double **X, int n, int d);

/*
 * Builds the normalized similarity matrix W = D^-1/2 * A * D^-1/2.
 * Because D is diagonal there is no need for real matrix products:
 * w_ij is simply a_ij / sqrt(d_i * d_j).
 * Input:  X - the n data points, each with d coordinates.
 * Output: a new n x n matrix W, or NULL if memory allocation failed.
 */
double **norm(double **X, int n, int d);

/*
 * Multiplies two matrices, C = A * B, and writes the result into C.
 * Input:  A - an r x m matrix, B - an m x c matrix,
 *         C - an existing r x c matrix that receives the result.
 */
void mat_mult(double **A, double **B, double **C, int r, int m, int c);

/*
 * Calculates H^T * H and writes the result into G.
 * Input:  H - an n x k matrix, G - an existing k x k matrix for the result.
 */
void gram(double **H, double **G, int n, int k);

/*
 * Allocates the helper matrices that update_step needs on every
 * iteration, so they are created once and not on every step.
 * Input:  buf - the struct to fill, n, k - the size of H.
 * Output: 1 on success, 0 if memory allocation failed (in that case
 *         nothing is left allocated).
 */
int alloc_buffers(UpdateBuffers *buf, int n, int k);

/*
 * Releases the helper matrices created by alloc_buffers.
 * Input:  buf - the struct to clean up.
 */
void free_buffers(UpdateBuffers *buf);

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
                   UpdateBuffers *buf, int n, int k);

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
                double eps);

/*
 * Reads every number from an open data file into one growing array.
 * The number of values on the first line tells us the dimension d.
 * Input:  fp - the open file, count and d - filled with the number of
 *         values read and the dimension.
 * Output: a new array with all the values, or NULL on failure.
 */
double *read_values(FILE *fp, int *count, int *d);

/*
 * Reads a data file (one comma-separated point per line) into a matrix.
 * Input:  path - the file to read, n and d - filled with the number of
 *         points and their dimension.
 * Output: a new n x d matrix, or NULL if anything went wrong.
 */
double **read_points(const char *path, int *n, int *d);

/*
 * Prints a matrix with each row on its own line, values separated by
 * commas and shown with 4 decimal places.
 * Input:  mat - the matrix, rows, cols - its size.
 */
void print_matrix(double **mat, int rows, int cols);

/*
 * Finds the function that matches the goal given on the command line.
 * Input:  goal - "sym", "ddg" or "norm".
 * Output: the matching function, or NULL if the goal is not valid.
 */
GoalFunc goal_function(const char *goal);

#endif
