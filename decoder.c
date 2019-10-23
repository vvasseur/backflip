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
#include <stdlib.h>
#include <string.h>

#include "decoder.h"
#include "param.h"
#include "sparse_cyclic.h"
#include "threshold.h"

static void fl_remove(fl_t fl, index_t pos);
static void fl_add(fl_t fl, index_t pos);
static void columns_to_rows(const sparse_t *restrict columns,
                            sparse_t *restrict rows);
static void compute_counters(const sparse_t *restrict columns,
                             const sparse_t *restrict rows,
                             const dense_t restrict checks,
                             dense_t *restrict counters);
static bit_t single_counter(const sparse_t restrict column, index_t position,
                            const dense_t restrict syndrome);
static void single_flip(const sparse_t restrict column, index_t position,
                        dense_t restrict syndrome);
static void compute_syndrome(decoder_t dec);

void alloc_decoder(decoder_t dec) {
    dec->bits = malloc(INDEX * sizeof(dense_t));
    dec->e = malloc(INDEX * sizeof(dense_t));
    dec->counters = malloc(INDEX * sizeof(bit_t *));
    for (index_t i = 0; i < INDEX; ++i) {
        dec->bits[i] = aligned_alloc(
            32, AVX_PADDING(2 * BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8);
        dec->e[i] = aligned_alloc(
            32, AVX_PADDING(2 * BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8);
        dec->counters[i] = aligned_alloc(
            32, AVX_PADDING(2 * BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8);
    }
    dec->syndrome = aligned_alloc(
        32, AVX_PADDING(2 * BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8);
    dec->Hrows = sparse_array_new(INDEX, BLOCK_WEIGHT);
    dec->fl = malloc(sizeof(struct flip_list));
    dec->fl->tod = malloc(INDEX * BLOCK_LENGTH * sizeof(((fl_t)0)->tod));
    dec->fl->next = malloc(INDEX * BLOCK_LENGTH * sizeof(((fl_t)0)->next));
    dec->fl->prev = malloc(INDEX * BLOCK_LENGTH * sizeof(((fl_t)0)->prev));
}

void free_decoder(decoder_t dec) {
    for (index_t i = 0; i < INDEX; ++i) {
        free(dec->bits[i]);
        free(dec->e[i]);
        free(dec->counters[i]);
    }
    free(dec->bits);
    free(dec->syndrome);
    free(dec->e);
    free(dec->counters);
    sparse_array_free(INDEX, dec->Hrows);
    free(dec->fl->tod);
    free(dec->fl->next);
    free(dec->fl->prev);
    free(dec->fl);
}

void reset_decoder(decoder_t dec) {
    memset(dec->syndrome, 0,
           AVX_PADDING(2 * BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8);
    for (index_t i = 0; i < INDEX; ++i) {
        memset(dec->bits[i], 0,
               AVX_PADDING(2 * BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8);
    }
    dec->fl->first = -1;
    dec->fl->length = 0;
}

static void fl_remove(fl_t fl, index_t pos) {
    index_t next = fl->next[pos];
    index_t prev = fl->prev[pos];
    if (next != -1) {
        fl->prev[next] = prev;
    }
    if (prev != -1) {
        fl->next[prev] = next;
    }
    else {
        fl->first = next;
    }
    --fl->length;
}

static void fl_add(fl_t fl, index_t pos) {
    fl->next[pos] = fl->first;
    fl->prev[pos] = -1;
    if (fl->first != -1)
        fl->prev[fl->first] = pos;
    fl->first = pos;
    ++fl->length;
}

void init_decoder_error(decoder_t dec, sparse_t *Hcolumns,
                        const sparse_t e_block, const sparse_t e2_block) {
    dec->Hcolumns = Hcolumns;
    columns_to_rows(Hcolumns, dec->Hrows);
    // dec->error_weight = ERROR_WEIGHT;

    for (index_t k = 0; k < INDEX; ++k) {
        for (index_t j = 0; j < BLOCK_LENGTH; ++j) {
            dec->e[k][j] = 0;
        }
    }
    index_t k;
    for (k = 0; k < ERROR_WEIGHT; ++k) {
        index_t j = e_block[k];
        if (j >= BLOCK_LENGTH)
            break;
        dec->e[0][j] = 1;
    }
    for (; k < ERROR_WEIGHT; ++k) {
        index_t j = e_block[k] - BLOCK_LENGTH;
        dec->e[1][j] = 1;
    }
    compute_syndrome(dec);

#if OUROBOROS
    if (e2_block) {
        for (index_t k = 0; k < SYNDROME_STOP; ++k) {
            dec->syndrome[e2_block[k]] ^= 1;
        }
    }
#endif
    dec->syndrome_weight = 0;
    for (index_t j = 0; j < BLOCK_LENGTH; ++j) {
        dec->syndrome_weight += dec->syndrome[j];
    }
}

static void columns_to_rows(const sparse_t *restrict columns,
                            sparse_t *restrict rows) {
    for (index_t i = 0; i < INDEX; ++i) {
        index_t l = 0;
        if ((*columns)[0] == 0) {
            (*rows)[0] = 0;
            l = 1;
        }
        else {
            (*rows)[0] = -(*columns)[BLOCK_WEIGHT - 1] + BLOCK_LENGTH;
        }
        for (index_t k = 1; k < BLOCK_WEIGHT; ++k) {
            (*rows)[k] = -(*columns)[BLOCK_WEIGHT + l - 1 - k] + BLOCK_LENGTH;
        }
        ++rows;
        ++columns;
    }
}

static void compute_counters(const sparse_t *restrict columns,
                             const sparse_t *restrict rows,
                             const dense_t restrict checks,
                             dense_t *restrict counters) {
    memcpy(checks + BLOCK_LENGTH, checks, BLOCK_LENGTH * sizeof(bit_t));
    for (index_t i = 0; i < INDEX; ++i) {
#ifndef AVX
        memset(counters[i], 0, BLOCK_LENGTH * sizeof(bit_t));
        multiply(BLOCK_LENGTH, BLOCK_WEIGHT, rows[i], checks, counters[i]);
#else
        multiply_avx2(AVX_PADDING(BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8,
                      BLOCK_WEIGHT, columns[i], checks, counters[i]);
#endif
    }
}

static bit_t single_counter(const sparse_t restrict column, index_t position,
                            const dense_t restrict syndrome) {
    bit_t counter = 0;
    index_t offset = position;

    index_t l;
    for (l = 0; l < BLOCK_WEIGHT; ++l) {
        index_t i = offset + column[l];
        if (i >= BLOCK_LENGTH) {
            offset -= BLOCK_LENGTH;
            break;
        }
        counter += syndrome[i];
    }
    for (; l < BLOCK_WEIGHT; ++l) {
        index_t i = offset + column[l];
        counter += syndrome[i];
    }
    return counter;
}

static void single_flip(const sparse_t restrict column, index_t position,
                        dense_t restrict syndrome) {
    index_t offset = position;

    index_t l;
    for (l = 0; l < BLOCK_WEIGHT; ++l) {
        index_t i = position + column[l];
        if (i >= BLOCK_LENGTH) {
            offset -= BLOCK_LENGTH;
            break;
        }
        syndrome[i] ^= 1;
    }
    for (; l < BLOCK_WEIGHT; ++l) {
        index_t i = offset + column[l];
        syndrome[i] ^= 1;
    }
}

#ifndef AVX
static void compute_syndrome(decoder_t dec) {
    for (index_t i = 0; i < INDEX; ++i) {
        multiply_mod2(BLOCK_LENGTH, BLOCK_WEIGHT, dec->Hcolumns[i], dec->e[i],
                      dec->syndrome);
    }
}
#else
static void compute_syndrome(decoder_t dec) {
    for (index_t i = 0; i < INDEX; ++i) {
        memcpy(dec->e[i] + BLOCK_LENGTH, dec->e[i],
               BLOCK_LENGTH * sizeof(bit_t));
        multiply_mod2_avx2(AVX_PADDING(BLOCK_LENGTH * 8 * sizeof(bit_t)) / 8,
                           BLOCK_WEIGHT, dec->Hrows[i], dec->e[i],
                           dec->syndrome);
    }
}
#endif

static inline int compute_ttl(int diff) {
    int ttl = (int)((diff)*TTL_COEFF0 + TTL_COEFF1);

    ttl = (ttl < 1) ? 1 : ttl;
    return (ttl > TTL_SATURATE) ? TTL_SATURATE : ttl;
}

int qcmdpc_decode_ttl(decoder_t dec, int max_iter) {
    dec->iter = 0;
    unsigned threshold;
    int recompute_threshold = 1;
    while (dec->iter < max_iter && dec->syndrome_weight != SYNDROME_STOP) {
        ++dec->iter;
        compute_counters(dec->Hcolumns, dec->Hrows, dec->syndrome,
                         dec->counters);
        if (recompute_threshold) {
            int t = ERROR_WEIGHT - dec->fl->length;
            t = (t > 0) ? t : 1;
            threshold = compute_threshold(dec->syndrome_weight, t);
            recompute_threshold = 0;
        }

        for (index_t k = 0; k < INDEX; ++k) {
            for (index_t j = 0; j < BLOCK_LENGTH; ++j) {
                if (dec->counters[k][j] >= threshold) {
                    recompute_threshold = 1;
                    if (dec->bits[k][j]) {
                        fl_remove(dec->fl, k * BLOCK_LENGTH + j);
                    }
                    else {
                        uint8_t ttl =
                            compute_ttl(dec->counters[k][j] - threshold);

                        fl_add(dec->fl, k * BLOCK_LENGTH + j);
                        dec->fl->tod[k * BLOCK_LENGTH + j] =
                            (dec->iter + ttl) % (TTL_SATURATE + 1);
                    }
                    bit_t counter =
                        single_counter(dec->Hcolumns[k], j, dec->syndrome);
                    single_flip(dec->Hcolumns[k], j, dec->syndrome);
                    dec->bits[k][j] ^= 1;
                    dec->syndrome_weight += BLOCK_WEIGHT - 2 * counter;
                    // dec->error_weight += 2 * (dec->bits[k][j] ^ dec->e[k][j])
                    // - 1;
                }
            }
        }
        if (dec->syndrome_weight != SYNDROME_STOP && dec->fl->length) {
            uint8_t current_iter = dec->iter % (TTL_SATURATE + 1);
            index_t fl_pos = dec->fl->first;
            while (fl_pos != -1) {
                if (dec->fl->tod[fl_pos] == current_iter) {
                    index_t k = 0;
                    index_t j = fl_pos;
                    if (fl_pos >= BLOCK_LENGTH) {
                        k = 1;
                        j -= BLOCK_LENGTH;
                    }

                    bit_t counter =
                        single_counter(dec->Hcolumns[k], j, dec->syndrome);
                    single_flip(dec->Hcolumns[k], j, dec->syndrome);
                    dec->bits[k][j] ^= 1;
                    dec->syndrome_weight += BLOCK_WEIGHT - 2 * counter;
                    // dec->error_weight += 2 * (dec->bits[k][j] ^ dec->e[k][j])
                    // - 1;
                    recompute_threshold = 1;

                    fl_remove(dec->fl, fl_pos);
                }
                fl_pos = dec->fl->next[fl_pos];
            }
        }
    }

    // return (!dec->error_weight);
    return (dec->syndrome_weight == SYNDROME_STOP);
}
