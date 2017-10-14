CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

word2vec : word2vec_original.c
	$(CC) word2vec_original.c -o word2vec $(CFLAGS)