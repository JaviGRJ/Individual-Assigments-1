#define _CRT_SECURE_NO_WARNINGS

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static void multiply(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double total = 0.0;
            for (size_t k = 0; k < n; ++k) {
                total += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = total;
        }
    }
}

static int equal(double actual, double expected) {
    const double absolute_tolerance = 1e-12;
    const double relative_tolerance = 1e-12;
    return fabs(actual - expected) <=
           absolute_tolerance + relative_tolerance * fabs(expected);
}

static int self_test(void) {
    const double a[] = {1, 2, 3, 4};
    const double b[] = {5, 6, 7, 8};
    const double expected[] = {19, 22, 43, 50};
    const double identity[] = {1, 0, 0, 1};
    const double zero[] = {0, 0, 0, 0};
    const double values[] = {2, -3, 4, 5};
    double result[4];

    multiply(a, b, result, 2);
    for (size_t i = 0; i < 4; ++i) {
        if (!equal(result[i], expected[i])) {
            fprintf(stderr, "known-answer self-test failed at element %zu\n", i);
            return 1;
        }
    }
    multiply(identity, values, result, 2);
    for (size_t i = 0; i < 4; ++i) {
        if (!equal(result[i], values[i])) {
            fprintf(stderr, "identity self-test failed at element %zu\n", i);
            return 1;
        }
    }
    multiply(zero, values, result, 2);
    for (size_t i = 0; i < 4; ++i) {
        if (!equal(result[i], 0.0)) {
            fprintf(stderr, "zero-matrix self-test failed at element %zu\n", i);
            return 1;
        }
    }
    return 0;
}

static int read_dataset(const char *path, double **a, double **b, size_t *n) {
    FILE *input = fopen(path, "r");
    if (input == NULL) {
        perror(path);
        return 1;
    }
    if (fscanf(input, "%zu", n) != 1 || *n == 0 || *n > 10000) {
        fprintf(stderr, "invalid matrix dimension in %s\n", path);
        fclose(input);
        return 1;
    }
    size_t count = *n * *n;
    *a = (double *)malloc(count * sizeof(double));
    *b = (double *)malloc(count * sizeof(double));
    if (*a == NULL || *b == NULL) {
        fprintf(stderr, "could not allocate input matrices\n");
        free(*a);
        free(*b);
        fclose(input);
        return 1;
    }
    for (size_t i = 0; i < count; ++i) {
        if (fscanf(input, "%lf", &(*a)[i]) != 1) {
            fprintf(stderr, "invalid matrix data in %s\n", path);
            free(*a);
            free(*b);
            fclose(input);
            return 1;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (fscanf(input, "%lf", &(*b)[i]) != 1) {
            fprintf(stderr, "invalid matrix data in %s\n", path);
            free(*a);
            free(*b);
            fclose(input);
            return 1;
        }
    }
    fclose(input);
    return 0;
}

static int write_result(const char *path, const double *c, size_t n) {
    FILE *output = fopen(path, "w");
    if (output == NULL) {
        perror(path);
        return 1;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (fprintf(output, j == 0 ? "%.17g" : " %.17g", c[i * n + j]) < 0) {
                fprintf(stderr, "failed while writing result\n");
                fclose(output);
                return 1;
            }
        }
        fputc('\n', output);
    }
    return fclose(output) != 0;
}

static double now_seconds(void) {
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    if (argc != 11 || strcmp(argv[1], "--input") != 0 ||
        strcmp(argv[3], "--output") != 0 || strcmp(argv[5], "--warmups") != 0 ||
        strcmp(argv[7], "--repetitions") != 0 || strcmp(argv[9], "--batch") != 0) {
        fprintf(stderr, "usage: matrix_multiply --input FILE --output FILE "
                        "--warmups N --repetitions N --batch N\n");
        return 2;
    }

    int warmups = atoi(argv[6]);
    int repetitions = atoi(argv[8]);
    int batch = atoi(argv[10]);
    if (warmups < 0 || repetitions < 1 || batch < 1) {
        fprintf(stderr, "warmups must be nonnegative; repetitions and batch positive\n");
        return 2;
    }
    double *a = NULL;
    double *b = NULL;
    size_t n = 0;
    if (read_dataset(argv[2], &a, &b, &n) != 0 || self_test() != 0) {
        free(a);
        free(b);
        return 1;
    }
    size_t count = n * n;
    double *c = (double *)malloc(count * sizeof(double));
    if (c == NULL) {
        fprintf(stderr, "could not allocate result matrix\n");
        free(a);
        free(b);
        return 1;
    }

    for (int repetition = 0; repetition < warmups; ++repetition) {
        multiply(a, b, c, n);
    }
    double *samples = (double *)malloc((size_t)repetitions * sizeof(double));
    if (samples == NULL) {
        fprintf(stderr, "could not allocate timing samples\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        double start = now_seconds();
        for (int run = 0; run < batch; ++run) {
            multiply(a, b, c, n);
        }
        samples[repetition] = (now_seconds() - start) / (double)batch;
    }
    if (write_result(argv[4], c, n) != 0) {
        free(a);
        free(b);
        free(c);
        free(samples);
        return 1;
    }

    double checksum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        checksum += c[i];
    }
    printf("{\"times_seconds\":[");
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        printf("%s%.17g", repetition == 0 ? "" : ",", samples[repetition]);
    }
    printf("],\"checksum\":%.17g,\"batch\":%d}\n", checksum, batch);
    free(a);
    free(b);
    free(c);
    free(samples);
    return 0;
}