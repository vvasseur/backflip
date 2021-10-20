# Backflip decoder

This is an implementation of the Backflip QC-MDPC decoder.

The Backflip algorithm relies on a ttl function that depends on the code
parameters, this program can be used to determine this function.

**Please use [this more up-to-date implementation which also includes other algorithms](https://github.com/vvasseur/qcmdpc_decoder) instead**.

## Usage

```sh
./qcmdpc_decoder_avx2 [OPTIONS]

-i, --max-iter         maximum number of iterations
-N, --rounds           number of rounds to perform
-T, --threads          number of threads to use
-q, --quiet            do not regularly output results (just on SIGHUP)
```

It generates QC-MDPC decoding instances then tries to decode them using the
Backflip algorithm.
For each instance, a random parity check matrix and a random error vector are
generated then the corresponding syndrome is computed.

Every 5 seconds, it prints the number of instances generated and the
distribution of the number of iterations it took to decode.

Unless a number of rounds is specified, it will only stop on SIGINT (Ctrl+C).


## Example

```sh
$ ./qcmdpc_decoder_avx2 -i6 -T8 -N100000
-DINDEX=2 -DBLOCK_LENGTH=32749 -DBLOCK_WEIGHT=137 -DERROR_WEIGHT=264 -DOUROBOROS=0 -DTTL_COEFF0=0.450000 -DTTL_COEFF1=1.000000 -DTTL_SATURATE=5
34107 3:68 4:24485 5:8930 6:598 >6:26
71056 3:128 4:51054 5:18616 6:1214 >6:44
100000 3:179 4:71898 5:26173 6:1688 >6:62
```

Out of 100000 instances, 179 were decoded in 3 iterations, 71898 in 4
iterations, 26173 in 5 iterations, 1688 in 6 iterations and 62 failed to decode
in at most 6 iterations.


## Parameters

Parameters are chosen at compile time. They are:
- `INDEX`
- `BLOCK_LENGTH`
- `BLOCK_WEIGHT`
- `ERROR_WEIGHT`
- `OUROBOROS` (0 or 1)
- `TTL_COEFF0`
- `TTL_COEFF1`
- `TTL_SATURATE`

For example:
```sh
$ EXTRA='-DINDEX=2 -DBLOCK_LENGTH=32749 -DBLOCK_WEIGHT=137 -DERROR_WEIGHT=264 -DOUROBOROS=0 -DTTL_COEFF0=0.435000 -DTTL_COEFF1=1.150000 -DTTL_SATURATE=5' make -B
```

Executable name is `qcmdpc_decoder_avx2`.


## Profile Guided Optimization

GCC does a good job at Profile Guided Optimization.
To use it, first compile with, for example:
```sh
$ EXTRA='-DINDEX=2 -DBLOCK_LENGTH=32749 -DBLOCK_WEIGHT=137 -DERROR_WEIGHT=264 -DOUROBOROS=0 -DTTL_COEFF0=0.435000 -DTTL_COEFF1=1.150000 -DTTL_SATURATE=5' make -B PROFGEN=1
```

Run the program on a sample with, for example (for 8 iterations, 8 threads and
a sample of size 100000):
```sh
$ ./qcmdpc_decoder_avx2 -i8 -T8 -N100000
```

Recompile to use PGO:
```sh
$ EXTRA='-DINDEX=2 -DBLOCK_LENGTH=32749 -DBLOCK_WEIGHT=137 -DERROR_WEIGHT=264 -DOUROBOROS=0 -DTTL_COEFF0=0.435000 -DTTL_COEFF1=1.150000 -DTTL_SATURATE=5' make -B PROFUSE=1
```

## AVX2

By default, the Makefile compiles the AVX2 version, if you do not have such an
instruction set build the 'noavx' target. Executable is then named
`qcmdpc_decoder`.


## Time-to-live function

The ttl function is chosen to be a saturating affine function.
The two parameters of that function are found by optimization.
We choose the function parameters that give the smallest DFR in simulation.

To optimize the ttl function for a set of parameters, use the `optimize-ttl.py`
python script. Its arguments are
`INDEX`, `BLOCK_LENGTH`, `BLOCK_WEIGHT`, `ERROR_WEIGHT`, `OUROBOROS`, `TTL_SATURATE`.
Additional arguments are directly passed to `qcmdpc_decoder_avx2`.

For example:
```sh
$ python3 optimize-select_avx2.py 2 32749 137 264 0 10 -i10 -T8
```
will optimize the ttl affine function for BIKE 1/2 IND-CPA Level 5 with 10
iterations running 8 threads.

Better results can sometimes be obtained by running it several times or editing
the `r0` value in the script.


# License

MIT (see file headers)
