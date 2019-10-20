/*
   Copyright (c) 2019 Valentin Vasseur

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE
*/
#include <omp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cli.h"
#include "decoder.h"
#include "param.h"
#include "sparse_cyclic.h"

/* In seconds */
#define TIME_BETWEEN_PRINTS 5

static void print_parameters(void);
static void print_stats(long int *n_test, long int *n_success);
static void inthandler(int signo);

static long int *n_test = NULL;
static long int *n_success = NULL;
static long int **n_iter = NULL;
static int n_threads = 1;
static int max_iter = 100;

static void print_parameters(void) {
    fprintf(stderr,
            "-DINDEX=%d "
            "-DBLOCK_LENGTH=%d "
            "-DBLOCK_WEIGHT=%d "
            "-DERROR_WEIGHT=%d "
            "-DOUROBOROS=%d "
            "-DTTL_COEFF0=%lf "
            "-DTTL_COEFF1=%lf "
            "-DTTL_SATURATE=%d\n",
            INDEX, BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT, OUROBOROS,
            TTL_COEFF0, TTL_COEFF1, TTL_SATURATE);
}

static void print_stats(long int *n_test, long int *n_success) {
    if (!n_test && !n_success)
        return;
    long int n_test_total = 0;
    long int n_success_total = 0;
    for (int i = 0; i < n_threads; ++i) {
        n_test_total += n_test[i];
        n_success_total += n_success[i];
    }
    long int n_iter_total[max_iter + 1];
    memset(n_iter_total, 0, (max_iter + 1) * sizeof(long int));

    for (int i = 0; i < n_threads; ++i) {
        for (int it = 0; it <= max_iter; ++it) {
            n_iter_total[it] += n_iter[i][it];
        }
    }

    fprintf(stderr, "%ld", n_test_total);
    for (int it = 0; it <= max_iter; ++it) {
        if (n_iter_total[it])
            fprintf(stderr, " %d:%ld", it, n_iter_total[it]);
    }
    if (n_success_total != n_test_total)
        fprintf(stderr, " >%d:%ld", max_iter, n_test_total - n_success_total);
    fprintf(stderr, "\n");
}

static void inthandler(int signo) {
    print_stats(n_test, n_success);

    if (signo != SIGHUP)
        exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]) {
    struct sigaction action;
    action.sa_handler = inthandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGHUP, &action, NULL);

    /* Number of test rounds */
    long int r = -1;
    /* PRNG seeds */
    uint64_t s[2] = {0, 0};
    int quiet = 0;

    parse_arguments(argc, argv, &max_iter, &r, &n_threads, &quiet);
    print_parameters();

    seed_random(&s[0], &s[1]);

    time_t last_print_time = time(NULL);

    /* Keep independent statistics for all threads. */
    n_test = calloc(n_threads, sizeof(long int));
    n_success = calloc(n_threads, sizeof(long int));
    n_iter = malloc(n_threads * sizeof(long int *));
    for (index_t i = 0; i < n_threads; ++i) {
        n_iter[i] = calloc(max_iter + 1, sizeof(long int));
    }

#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();

        int thread_quiet = tid ? 1 : quiet;

        /* Parity check matrix */
        sparse_t *H = sparse_array_new(INDEX, BLOCK_WEIGHT);
        /* Error pattern */
        sparse_t e_block = sparse_new(ERROR_WEIGHT);

        /* Error pattern on the syndrome (for Ouroboros) */
#if !OUROBOROS
        sparse_t e2_block = NULL;
#else
        sparst_t e2_block = sparse_new(ERROR_WEIGHT / 2);
#endif

        struct decoder dec;
        alloc_decoder(&dec);

        prng_t prng = malloc(sizeof(struct PRNG));
        prng->s0 = s[0];
        prng->s1 = s[1];
        prng->random_lim = random_lim;
        prng->random_uint64_t = random_uint64_t;

        for (int i = 0; i < tid; ++i) {
            jump(&prng->s0, &prng->s1);
        }

        long int thread_total_tests = (tid + r) / n_threads;
        while (r == -1 || n_test[tid] < thread_total_tests) {
            sparse_array_rand(INDEX, BLOCK_LENGTH, BLOCK_WEIGHT, prng, H);

            sparse_rand(INDEX * BLOCK_LENGTH, ERROR_WEIGHT, prng, e_block);
#if OUROBOROS
            sparse_rand(BLOCK_LENGTH, SYNDROME_STOP, prng, e2_block);
#endif

            reset_decoder(&dec);
            init_decoder_error(&dec, H, e_block, e2_block);

            if (qcmdpc_decode_ttl(&dec, max_iter)) {
                n_success[tid]++;
                n_iter[tid][dec.iter]++;
            }

            n_test[tid]++;

            time_t current_time;
            if (!thread_quiet && (current_time = time(NULL)) >
                                     last_print_time + TIME_BETWEEN_PRINTS) {
                print_stats(n_test, n_success);
                last_print_time = current_time;
            }
        }
        free(prng);
        sparse_array_free(INDEX, H);
        sparse_free(e_block);
        if (e2_block) {
            sparse_free(e2_block);
        }
        free_decoder(&dec);
    }

    print_stats(n_test, n_success);
    free(n_test);
    free(n_success);
    for (index_t i = 0; i < n_threads; ++i) {
        free(n_iter[i]);
    }
    free(n_iter);
    exit(EXIT_SUCCESS);
}
