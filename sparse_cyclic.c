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
#ifdef AVX
#include <immintrin.h>
#endif
#include <stdlib.h>

#include "sparse_cyclic.h"

static void insert_sorted(index_t value, index_t max_i, index_t *array);

sparse_t sparse_new(index_t weight) {
    sparse_t h = (index_t *)malloc(weight * sizeof(index_t));

    return h;
}

void sparse_free(sparse_t h) { free(h); }

sparse_t *sparse_array_new(index_t index, index_t weight) {
    sparse_t *h = malloc(index * sizeof(sparse_t));

    for (index_t i = 0; i < index; ++i) {
        h[i] = sparse_new(weight);
    }

    return h;
}

void sparse_array_free(index_t index, sparse_t *h) {
    for (index_t i = 0; i < index; ++i) {
        sparse_free(h[i]);
    }
    free(h);
}

/* Insert in place. */
static void insert_sorted(index_t value, index_t max_i, index_t *array) {
    index_t i;
    for (i = 0; i < max_i && array[i] <= value; i++, value++)
        ;
    for (index_t j = max_i; j > i; j--)
        array[j] = array[j - 1];
    array[i] = value;
}

/* Pick a random (sparse) binary block h of weight 'weight' in a previously
 * allocated block. */
sparse_t sparse_rand(index_t length, index_t weight, prng_t prng, sparse_t h) {
    /* Get an ordered list of positions for which the bit should be set to 1. */
    for (index_t i = 0; i < weight; i++) {
        index_t rand = prng->random_lim(--length, &prng->s0, &prng->s1);
        insert_sorted(rand, i, h);
    }
    return h;
}

sparse_t *sparse_array_rand(index_t index, index_t length, index_t weight,
                            prng_t prng, sparse_t *H) {
    for (index_t i = 0; i < index; ++i) {
        sparse_rand(length, weight, prng, H[i]);
    }

    return H;
}

struct mult_t {
    dense_t y;
    dense_t z;
    index_t len;
};

void multiply_mod2(index_t block_length, index_t block_weight,
                   const sparse_t restrict x, const dense_t restrict y,
                   dense_t restrict z) {
    /* Avoid doing a modulo operation by precomputing the wrapping around. */
    struct mult_t queue[2 * block_weight];
    struct mult_t *restrict queue1 = queue;
    struct mult_t *restrict queue2 = queue + block_weight;
    for (index_t k = 0; k < block_weight; ++k) {
        queue1[k].z = z + x[k];
        queue1[k].y = y;
        queue1[k].len = block_length - x[k];
        queue2[k].z = z;
        queue2[k].y = y + block_length - x[k];
        queue2[k].len = x[k];
    }
    for (index_t k = 0; k < 2 * block_weight; ++k) {
        dense_t restrict y = queue[k].y;
        dense_t restrict z = queue[k].z;
        index_t len = queue[k].len;

        for (index_t i = 0; i < len; ++i) {
            z[i] ^= y[i];
        }
    }
}

void multiply(index_t block_length, index_t block_weight,
              const sparse_t restrict x, const dense_t restrict y,
              dense_t restrict z) {
    /* Avoid doing a modulo operation by precomputing the wrapping around. */
    struct mult_t queue[2 * block_weight];
    struct mult_t *restrict queue1 = queue;
    struct mult_t *restrict queue2 = queue + block_weight;
    for (index_t k = 0; k < block_weight; ++k) {
        queue1[k].z = z + x[k];
        queue1[k].y = y;
        queue1[k].len = block_length - x[k];
        queue2[k].z = z;
        queue2[k].y = y + block_length - x[k];
        queue2[k].len = x[k];
    }
    for (index_t k = 0; k < 2 * block_weight; ++k) {
        dense_t restrict y = queue[k].y;
        dense_t restrict z = queue[k].z;
        index_t len = queue[k].len;

        for (index_t i = 0; i < len; ++i) {
            z[i] += y[i];
        }
    }
}

#ifdef AVX
void multiply_mod2_avx2(index_t block_length, index_t block_weight,
                        const sparse_t restrict x, const dense_t restrict y,
                        dense_t restrict z) {
    /* I could not manage to make GCC unroll that loop automatically. */
    for (index_t i = 0; i < block_length / 32; i += 16) {
        __m256i vec_x0;
        __m256i vec_x1;
        __m256i vec_x2;
        __m256i vec_x3;
        __m256i vec_x4;
        __m256i vec_x5;
        __m256i vec_x6;
        __m256i vec_x7;
        __m256i vec_x8;
        __m256i vec_x9;
        __m256i vec_x10;
        __m256i vec_x11;
        __m256i vec_x12;
        __m256i vec_x13;
        __m256i vec_x14;
        __m256i vec_x15;
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x0)
                     : [ z ] "m"(((__m256i *)z)[i])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x1)
                     : [ z ] "m"(((__m256i *)z)[i + 1])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x2)
                     : [ z ] "m"(((__m256i *)z)[i + 2])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x3)
                     : [ z ] "m"(((__m256i *)z)[i + 3])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x4)
                     : [ z ] "m"(((__m256i *)z)[i + 4])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x5)
                     : [ z ] "m"(((__m256i *)z)[i + 5])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x6)
                     : [ z ] "m"(((__m256i *)z)[i + 6])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x7)
                     : [ z ] "m"(((__m256i *)z)[i + 7])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x8)
                     : [ z ] "m"(((__m256i *)z)[i + 8])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x9)
                     : [ z ] "m"(((__m256i *)z)[i + 9])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x10)
                     : [ z ] "m"(((__m256i *)z)[i + 10])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x11)
                     : [ z ] "m"(((__m256i *)z)[i + 11])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x12)
                     : [ z ] "m"(((__m256i *)z)[i + 12])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x13)
                     : [ z ] "m"(((__m256i *)z)[i + 13])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x14)
                     : [ z ] "m"(((__m256i *)z)[i + 14])
                     :);
        asm volatile("vmovdqa   %[z], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x15)
                     : [ z ] "m"(((__m256i *)z)[i + 15])
                     :);

        for (index_t j = 0; j < block_weight; ++j) {
            index_t off = x[j] + i * 32;
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x0)
                         : [ y ] "m"(*(__m256i *)&y[x[j] + i * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x1)
                         : [ y ] "m"(*(__m256i *)&y[off + 1 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x2)
                         : [ y ] "m"(*(__m256i *)&y[off + 2 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x3)
                         : [ y ] "m"(*(__m256i *)&y[off + 3 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x4)
                         : [ y ] "m"(*(__m256i *)&y[off + 4 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x5)
                         : [ y ] "m"(*(__m256i *)&y[off + 5 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x6)
                         : [ y ] "m"(*(__m256i *)&y[off + 6 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x7)
                         : [ y ] "m"(*(__m256i *)&y[off + 7 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x8)
                         : [ y ] "m"(*(__m256i *)&y[off + 8 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x9)
                         : [ y ] "m"(*(__m256i *)&y[off + 9 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x10)
                         : [ y ] "m"(*(__m256i *)&y[off + 10 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x11)
                         : [ y ] "m"(*(__m256i *)&y[off + 11 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x12)
                         : [ y ] "m"(*(__m256i *)&y[off + 12 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x13)
                         : [ y ] "m"(*(__m256i *)&y[off + 13 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x14)
                         : [ y ] "m"(*(__m256i *)&y[off + 14 * 32])
                         :);
            asm volatile("vpxor   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x15)
                         : [ y ] "m"(*(__m256i *)&y[off + 15 * 32])
                         :);
        }
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i])
                     : [ vec_x ] "x"(vec_x0)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 1])
                     : [ vec_x ] "x"(vec_x1)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 2])
                     : [ vec_x ] "x"(vec_x2)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 3])
                     : [ vec_x ] "x"(vec_x3)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 4])
                     : [ vec_x ] "x"(vec_x4)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 5])
                     : [ vec_x ] "x"(vec_x5)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 6])
                     : [ vec_x ] "x"(vec_x6)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 7])
                     : [ vec_x ] "x"(vec_x7)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 8])
                     : [ vec_x ] "x"(vec_x8)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 9])
                     : [ vec_x ] "x"(vec_x9)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 10])
                     : [ vec_x ] "x"(vec_x10)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 11])
                     : [ vec_x ] "x"(vec_x11)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 12])
                     : [ vec_x ] "x"(vec_x12)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 13])
                     : [ vec_x ] "x"(vec_x13)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 14])
                     : [ vec_x ] "x"(vec_x14)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 15])
                     : [ vec_x ] "x"(vec_x15)
                     :);
    }
}

void multiply_avx2(index_t block_length, index_t block_weight,
                   const sparse_t restrict x, const dense_t restrict y,
                   dense_t restrict z) {
    /* I could not manage to make GCC unroll that loop automatically. */
    for (index_t i = 0; i < block_length / 32; i += 16) {
        __m256i vec_x0;
        __m256i vec_x1;
        __m256i vec_x2;
        __m256i vec_x3;
        __m256i vec_x4;
        __m256i vec_x5;
        __m256i vec_x6;
        __m256i vec_x7;
        __m256i vec_x8;
        __m256i vec_x9;
        __m256i vec_x10;
        __m256i vec_x11;
        __m256i vec_x12;
        __m256i vec_x13;
        __m256i vec_x14;
        __m256i vec_x15;
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x0));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x1));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x2));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x3));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x4));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x5));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x6));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x7));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x8));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x9));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x10));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x11));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x12));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x13));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x14));
        asm volatile("vpxor   %[vec_x], %[vec_x], %[vec_x]\n\t"
                     : [ vec_x ] "=x"(vec_x15));

        for (index_t j = 0; j < block_weight; ++j) {
            index_t off = x[j] + i * 32;
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x0)
                         : [ y ] "m"(*(__m256i *)&y[off])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x1)
                         : [ y ] "m"(*(__m256i *)&y[off + 1 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x2)
                         : [ y ] "m"(*(__m256i *)&y[off + 2 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x3)
                         : [ y ] "m"(*(__m256i *)&y[off + 3 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x4)
                         : [ y ] "m"(*(__m256i *)&y[off + 4 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x5)
                         : [ y ] "m"(*(__m256i *)&y[off + 5 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x6)
                         : [ y ] "m"(*(__m256i *)&y[off + 6 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x7)
                         : [ y ] "m"(*(__m256i *)&y[off + 7 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x8)
                         : [ y ] "m"(*(__m256i *)&y[off + 8 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x9)
                         : [ y ] "m"(*(__m256i *)&y[off + 9 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x10)
                         : [ y ] "m"(*(__m256i *)&y[off + 10 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x11)
                         : [ y ] "m"(*(__m256i *)&y[off + 11 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x12)
                         : [ y ] "m"(*(__m256i *)&y[off + 12 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x13)
                         : [ y ] "m"(*(__m256i *)&y[off + 13 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x14)
                         : [ y ] "m"(*(__m256i *)&y[off + 14 * 32])
                         :);
            asm volatile("vpaddb   %[y], %[vec_x], %[vec_x]\n\t"
                         : [ vec_x ] "+x"(vec_x15)
                         : [ y ] "m"(*(__m256i *)&y[off + 15 * 32])
                         :);
        }
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i])
                     : [ vec_x ] "x"(vec_x0)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 1])
                     : [ vec_x ] "x"(vec_x1)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 2])
                     : [ vec_x ] "x"(vec_x2)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 3])
                     : [ vec_x ] "x"(vec_x3)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 4])
                     : [ vec_x ] "x"(vec_x4)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 5])
                     : [ vec_x ] "x"(vec_x5)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 6])
                     : [ vec_x ] "x"(vec_x6)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 7])
                     : [ vec_x ] "x"(vec_x7)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 8])
                     : [ vec_x ] "x"(vec_x8)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 9])
                     : [ vec_x ] "x"(vec_x9)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 10])
                     : [ vec_x ] "x"(vec_x10)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 11])
                     : [ vec_x ] "x"(vec_x11)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 12])
                     : [ vec_x ] "x"(vec_x12)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 13])
                     : [ vec_x ] "x"(vec_x13)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 14])
                     : [ vec_x ] "x"(vec_x14)
                     :);
        asm volatile("vmovdqa   %[vec_x], %[z]\n\t"
                     : [ z ] "=m"(((__m256i *)z)[i + 15])
                     : [ vec_x ] "x"(vec_x15)
                     :);
    }
}
#endif
