#!/usr/bin/python3
# Copyright (c) 2019 Valentin Vasseur
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE
from subprocess import run, PIPE, Popen
from scipy.optimize import minimize
from math import sqrt, log10, floor
from scipy.special import erfinv
import re
import random
import sys
import os
import signal
import time

WAIT_TIME = 5
MIN_TEST = 1000
COV = 0.95
PREC = 1e-1

if len(sys.argv) < 7:
    print(
        "Usage: INDEX BLOCK_LENGTH BLOCK_WEIGHT ERROR_WEIGHT OUROBOROS TTL_SATURATE",
        file=sys.stderr)
INDEX = int(sys.argv[1])
BLOCK_LENGTH = int(sys.argv[2])
BLOCK_WEIGHT = int(sys.argv[3])
ERROR_WEIGHT = int(sys.argv[4])
OUROBOROS = int(sys.argv[5])
TTL_SATURATE = int(sys.argv[6])
COMPILER_OPTIONS = "-DINDEX={} -DBLOCK_LENGTH={} -DBLOCK_WEIGHT={} -DERROR_WEIGHT={} -DOUROBOROS={} -DTTL_SATURATE={}".format(
    INDEX, BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT, OUROBOROS, TTL_SATURATE)
PROGRAM_OPTIONS = ['-q'] + sys.argv[7:]

PROGRAM_PATH = "./qcmdpc_decoder"
# Comment if noavx was built
PROGRAM_PATH += "_avx2"

PGO = ""
# Uncomment if a profile has been generated.
#PGO += " PROFUSE=1"

DEPEND_ON_PARAM_H = "decoder.o qcmdpc_decoder.o threshold.o"

proc = None

# Start the optimization with an initial simplex consisting of points around
# r0, randomly spread with weights coeffs
r0 = [1, 1.5]
coeffs = [0.75, 0.95]
initial_simplex = [[
    a + c * (1 - 2 * random.random()) for a, c in zip(r0, coeffs)
] for i in range(len(r0) + 1)]

# Store the DFR corresponding to a ttl function (keys of this dictionary are
# the images of the ttl function on the relevant domain)
dyn_dfr = {}


# Without this signal handling, the decoder would continue running in the
# background if the python script is exited with CTRL-C
def signal_handler(sig, frame):
    if proc:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Compute the image of the ttl function on the relevant interval
def ttl_function(a, b):
    def ttl(i):
        res = floor(a * i + b)
        if res < 1:
            res = 1
        if res > TTL_SATURATE:
            res = TTL_SATURATE
        return res

    values = [ttl(i) for i in range(0, BLOCK_WEIGHT // 2 + 1)]
    return tuple(values)


# Compute Wilson score interval
def wilson(dfr, nb_test, coverage):
    z = sqrt(2) * erfinv(coverage)
    if nb_test <= 0:
        return 0, 1
    if z * z - 1 / nb_test + 4 * nb_test * dfr * (1 - dfr) + (4 * dfr - 2) < 0:
        return 0, 1
    w_minus = max(0, (2 * nb_test * dfr + z * z -
                      z * sqrt(z * z - 1 / nb_test + 4 * nb_test * dfr *
                               (1 - dfr) + (4 * dfr - 2)) + 1) /
                  (2 * (nb_test + z * z)))
    w_plus = min(1, (2 * nb_test * dfr + z * z +
                     z * sqrt(z * z - 1 / nb_test + 4 * nb_test * dfr *
                              (1 - dfr) - (4 * dfr - 2)) + 1) /
                 (2 * (nb_test + z * z)))
    if w_minus == 0:
        return w_minus, w_plus

    return w_minus, w_plus


dfr_min_plus = 1.


def dfr_comp(r):
    global PREC
    global proc
    global dfr_min_plus
    orig_nb_test = 0
    orig_nb_fail = 0
    nb_test = 0
    nb_fail = 0
    last_nb_fail = 0
    dfr = 1.
    ttl_values = ttl_function(*r)

    # Do not recompute DFR for a ttl function that has already been tested
    if tuple(ttl_values) in dyn_dfr:
        orig_nb_test, orig_nb_fail = dyn_dfr[tuple(ttl_values)]
        dfr = orig_nb_fail / orig_nb_test
        w_minus, w_plus = wilson(dfr, orig_nb_test, COV)
        amplitude_wilson = w_plus - w_minus
        if amplitude_wilson < PREC and orig_nb_test > MIN_TEST:
            print(" ".join(["{!r}" for i in range(len(r) + 3)
                            ]).format(*r, dfr, orig_nb_test, orig_nb_fail))
            print(
                "--------------------------------------------------------------------------------"
            )
            return dfr

    # Recompile and run
    command = "EXTRA=\"{} {}\" ".format(
        COMPILER_OPTIONS,
        " ".join(["-D'" + "TTL_COEFF{}'={}"
                  for i in r]).format(*list(sum(enumerate(r), ()))))
    command += "make -s -B {} {};".format(DEPEND_ON_PARAM_H, PGO)
    command += "make -s {};".format(PROGRAM_PATH, PGO)
    command += "{} {}".format(PROGRAM_PATH, " ".join(PROGRAM_OPTIONS))
    proc = Popen(command, shell=True, stderr=PIPE, preexec_fn=os.setsid)
    out = proc.stderr
    for l in out:
        l = l.decode("utf-8").strip('\n')
        match = re.match("([0-9]+) .* (>[0-9]+:([0-9]+))?", l)
        if not match:
            print(l)
            time.sleep(WAIT_TIME)
            os.killpg(os.getpgid(proc.pid), signal.SIGHUP)
            continue

        nb_test = int(match.group(1))
        if match.group(3):
            nb_fail = int(match.group(3))
        else:
            nb_fail = 0

        if nb_test == 0:
            time.sleep(WAIT_TIME)
            os.killpg(os.getpgid(proc.pid), signal.SIGHUP)
            continue

        dfr = nb_fail / nb_test
        w_minus, w_plus = wilson(dfr, nb_test, COV)
        if w_minus <= 0 or w_plus <= 0:
            time.sleep(WAIT_TIME)
            os.killpg(os.getpgid(proc.pid), signal.SIGHUP)
            continue

        amplitude_wilson = log10(w_plus) - log10(w_minus)

        print(l + " {}".format(amplitude_wilson), file=sys.stderr)

        if amplitude_wilson > PREC:
            last_nb_fail = nb_fail

        if amplitude_wilson < PREC and nb_test > MIN_TEST:
            if w_plus < dfr_min_plus:
                dfr_min_plus = w_plus
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        elif w_minus > dfr_min_plus:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            time.sleep(WAIT_TIME)
        os.killpg(os.getpgid(proc.pid), signal.SIGHUP)

    print(" ".join(["{!r}" for i in range(len(r) + 3)
                    ]).format(*r, dfr, nb_test, nb_fail))
    print(
        "--------------------------------------------------------------------------------"
    )
    dyn_dfr[ttl_values] = nb_test, nb_fail
    return dfr


print(" ".join(PROGRAM_OPTIONS))

res = minimize(dfr_comp,
               r0,
               method='Nelder-Mead',
               options={
                   'adaptive': True,
                   'initial_simplex': initial_simplex
               })

print("-DTTL_COEFF0={} -DTTL_COEFF1={}".format(*res['x']))
