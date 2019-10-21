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
#include <math.h>

#include "param.h"
#include "threshold.h"

static double lnbino(unsigned n, unsigned t);
static double xlny(double x, double y);
static double lnbinomialpmf(unsigned n, unsigned k, double p, double q);
static double Euh_log(unsigned t, unsigned i);
static double iks(unsigned t);
static double counters_C0(unsigned S, unsigned t, double x);
static double counters_C1(unsigned S, unsigned t, double x);

static double lnbino(unsigned n, unsigned t) {
    if ((t == 0) || (n == t))
        return 0.0;
    else
        return lgamma(n + 1) - lgamma(t + 1) - lgamma(n - t + 1);
}

static double xlny(double x, double y) {
    if (x == 0.)
        return 0.;
    else
        return x * log(y);
}

/* Log of the probability mass function of a binomial distribution */
static double lnbinomialpmf(unsigned n, unsigned k, double p, double q) {
    return lnbino(n, k) + xlny(k, p) + xlny(n - k, q);
}

static double Euh_log(unsigned t, unsigned i) {
    return lnbino(INDEX * BLOCK_WEIGHT, i) +
           lnbino(INDEX * (BLOCK_LENGTH - BLOCK_WEIGHT), t - i) -
           lnbino(INDEX * BLOCK_LENGTH, t);
}

/* iks = X = sum((l - 1) * E_l, l odd) */
static double iks(unsigned t) {
    unsigned i;
    double x;
    double denom = 0.;

    /* Euh_log(n, w, t, i) decreases fast when 'i' varies.
     For i >= 10 it is very likely to be negligible. */
    for (x = 0, i = 1; (i < 10) && (i < t); i += 2) {
        x += (i - 1) * exp(Euh_log(t, i));
        denom += exp(Euh_log(t, i));
    }

    if (denom == 0.)
        return 0.;
    return x / denom;
}

/* Probability for a bit of the syndrome to be zero, knowing the syndrome
 * weight 'S' and 'X'. */
static double counters_C0(unsigned S, unsigned t, double x) {
    return ((INDEX * BLOCK_WEIGHT - 1) * S - x) / (INDEX * BLOCK_LENGTH - t) /
           BLOCK_WEIGHT;
}

/* Probability for a bit of the syndrome to be non-zero, knowing the syndrome
 * weight 'S' and 'X'. */
static double counters_C1(unsigned S, unsigned t, double x) {
    return (S + x) / t / BLOCK_WEIGHT;
}

unsigned compute_threshold(unsigned S, unsigned t) {
    double p, q;

    double x = iks(t) * S;
    p = counters_C0(S, t, x);
    q = counters_C1(S, t, x);

    unsigned threshold;
    if (p >= 1.0 || p > q) {
        threshold = BLOCK_WEIGHT;
    }
    else if (q >= 1.) {
        threshold = BLOCK_WEIGHT + 1;
        double diff = 0.;
        do {
            threshold--;
            diff = -exp(lnbinomialpmf(BLOCK_WEIGHT, threshold, p, 1. - p)) *
                       (INDEX * BLOCK_LENGTH - t) +
                   1.;
        } while (diff >= 0. && threshold > (BLOCK_WEIGHT + 1) / 2);
        threshold = threshold < BLOCK_WEIGHT ? (threshold + 1) : BLOCK_WEIGHT;
    }
    else {
        threshold = BLOCK_WEIGHT + 1;
        double diff = 0.;
        do {
            threshold--;
            diff = (-exp(lnbinomialpmf(BLOCK_WEIGHT, threshold, p, 1. - p)) *
                        (INDEX * BLOCK_LENGTH - t) +
                    exp(lnbinomialpmf(BLOCK_WEIGHT, threshold, q, 1. - q)) * t);
        } while (diff >= 0. && threshold > (BLOCK_WEIGHT + 1) / 2);
        threshold = threshold < BLOCK_WEIGHT ? (threshold + 1) : BLOCK_WEIGHT;
    }

    return threshold;
}
