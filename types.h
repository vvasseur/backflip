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
#ifndef TYPES_H
#define TYPES_H
#include <stdint.h>

/* Round relevant arrays size to the next multiple of 16 * 256 bits (to use the
 * 16 ymm AVX registers). */
#define AVX_PADDING(len) ((len + (256 * 16) - 1) / (256 * 16)) * (256 * 16)

typedef int_fast32_t index_t;
typedef index_t *sparse_t;

typedef uint8_t bit_t;
typedef bit_t *dense_t;

typedef struct ring_buffer *ring_buffer_t;
typedef struct flip_list *fl_t;
typedef struct parameters *parameters_t;
typedef struct decoder *decoder_t;

/* Double linked list to store previous flips */
struct flip_list {
    index_t first;
    uint8_t *tod;
    index_t *prev;
    index_t *next;
    index_t length;
};

/* State of the decoder */
struct decoder {
    sparse_t *Hcolumns;
    sparse_t *Hrows;
    dense_t *bits;
    dense_t syndrome;
    dense_t *e;
    bit_t **counters;
    fl_t fl;
    index_t syndrome_weight;
    // index_t error_weight;
    index_t iter;
};
#endif
