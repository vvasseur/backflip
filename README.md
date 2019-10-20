# Backflip decoder

This is an implementation of the Backflip QC-MDPC decoder.

The Backflip algorithm relies on a ttl function that depend on the code
parameters, this program can be used to determine this function.


## Quickstart

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
will optimize the ttl affine function for BIKE IND-CPA Level 5 with 10
iterations running 8 threads.

Better results can sometimes be obtained by running it several times or editing
the `r0` value in the script.


# License

MIT (see file headers)
