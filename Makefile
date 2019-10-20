CC=gcc
SRC=cli.c decoder.c qcmdpc_decoder.c sparse_cyclic.c threshold.c xoroshiro128plus.c
OBJ=$(SRC:%.c=%.o)
DEP=$(SRC:%.c=%.d)
LFLAGS=-lm
CFLAGS=-Wall -std=gnu11 $(OPT) $(EXTRA)
ifdef AVX
    CFLAGS+=-DAVX
endif
ifdef PROFGEN
    CFLAGS+=-fprofile-generate
endif
ifdef PROFUSE
    CFLAGS+=-fprofile-use -fprofile-correction
endif

default: avx2

noavx:
	make OPT="-Ofast -march=native -flto" qcmdpc_decoder

avx2:
	make OPT="-Ofast -march=native -flto" AVX=1 qcmdpc_decoder_avx2

format:
	clang-format -i -style=file *.c *.h

qcmdpc_decoder: $(OBJ)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LFLAGS)

qcmdpc_decoder_avx2: $(OBJ)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LFLAGS)

qcmdpc_decoder.o: qcmdpc_decoder.c
	$(CC) $(CFLAGS) -MMD -fopenmp -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -MMD -c -o $@ $<

-include $(DEP)

clean:
	- /bin/rm qcmdpc_decoder qcmdpc_decoder_avx2 $(OBJ) $(DEP)
