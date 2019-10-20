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
#ifndef SPARSE_CYCLIC_H
#define SPARSE_CYCLIC_H
#include "types.h"
#include "xoroshiro128plus.h"

sparse_t sparse_new(index_t weight);
void sparse_free(sparse_t h);
sparse_t sparse_rand(index_t length, index_t weight, prng_t prng, sparse_t h);

sparse_t *sparse_array_new(index_t index, index_t weight);
void sparse_array_free(index_t index, sparse_t *h);
sparse_t *sparse_array_rand(index_t index, index_t length, index_t weight,
                            prng_t prng, sparse_t *H);

void multiply(index_t block_length, index_t block_weight,
              const sparse_t restrict x, const dense_t restrict y,
              dense_t restrict z);
void multiply_mod2(index_t block_length, index_t block_weight,
                   const sparse_t restrict x, const dense_t restrict y,
                   dense_t restrict z);
#ifdef AVX
void multiply_avx2(index_t block_length, index_t block_weight,
                   const sparse_t restrict x, const dense_t restrict y,
                   dense_t restrict z);
void multiply_mod2_avx2(index_t block_length, index_t block_weight,
                        const sparse_t restrict x, const dense_t restrict y,
                        dense_t restrict z);
#endif
#endif
