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
#ifndef PARAM_H
#define PARAM_H
#if !defined(PRESET) && !(defined(INDEX) && defined(BLOCK_LENGTH) &&           \
                          defined(BLOCK_WEIGHT) && defined(ERROR_WEIGHT))
#define PRESET 256
#endif
#ifndef OUROBOROS
#define OUROBOROS 0
#endif

#ifdef PRESET
#if PRESET == 128 && !OUROBOROS
#ifndef INDEX
#define INDEX 2
#endif
#ifndef BLOCK_LENGTH
#define BLOCK_LENGTH 10163
#endif
#ifndef ERROR_WEIGHT
#define ERROR_WEIGHT 134
#endif
#ifndef BLOCK_WEIGHT
#define BLOCK_WEIGHT 71
#endif
#elif PRESET == 128 && OUROBOROS
#ifndef INDEX
#define INDEX 2
#endif
#ifndef BLOCK_LENGTH
#define BLOCK_LENGTH 11027
#endif
#ifndef ERROR_WEIGHT
#define ERROR_WEIGHT 156
#endif
#ifndef BLOCK_WEIGHT
#define BLOCK_WEIGHT 67
#endif
#elif PRESET == 192 && !OUROBOROS
#ifndef INDEX
#define INDEX 2
#endif
#ifndef BLOCK_LENGTH
#define BLOCK_LENGTH 19853
#endif
#ifndef ERROR_WEIGHT
#define ERROR_WEIGHT 199
#endif
#ifndef BLOCK_WEIGHT
#define BLOCK_WEIGHT 103
#endif
#elif PRESET == 192 && OUROBOROS
#ifndef INDEX
#define INDEX 2
#endif
#ifndef BLOCK_LENGTH
#define BLOCK_LENGTH 21683
#endif
#ifndef ERROR_WEIGHT
#define ERROR_WEIGHT 226
#endif
#ifndef BLOCK_WEIGHT
#define BLOCK_WEIGHT 99
#endif
#elif PRESET == 256 && !OUROBOROS
#ifndef INDEX
#define INDEX 2
#endif
#ifndef BLOCK_LENGTH
#define BLOCK_LENGTH 32749
#endif
#ifndef ERROR_WEIGHT
#define ERROR_WEIGHT 264
#endif
#ifndef BLOCK_WEIGHT
#define BLOCK_WEIGHT 137
#endif
#elif PRESET == 256 && OUROBOROS
#ifndef INDEX
#define INDEX 2
#endif
#ifndef BLOCK_LENGTH
#define BLOCK_LENGTH 36131
#endif
#ifndef ERROR_WEIGHT
#define ERROR_WEIGHT 300
#endif
#ifndef BLOCK_WEIGHT
#define BLOCK_WEIGHT 133
#endif
#endif
#endif

#if OUROBOROS && !defined(SYNDROME_STOP)
#define SYNDROME_STOP ((ERROR_WEIGHT) / 2)
#else
#define SYNDROME_STOP 0
#endif

#ifndef TTL_COEFF0
#define TTL_COEFF0 0.435
#endif
#ifndef TTL_COEFF1
#define TTL_COEFF1 1.15
#endif
#ifndef TTL_SATURATE
#define TTL_SATURATE 5
#endif

#if INDEX != 2
#error "INDEX != 2: Not implemented"
#endif
#if BLOCK_WEIGHT > 255
#error "BLOCK_WEIGHT > 255: Not implemented"
#endif
#if BLOCK_LENGTH > 65536
#error "BLOCK_LENGTH > 65536: Not implemented"
#endif
#endif
