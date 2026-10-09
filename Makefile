CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

symnmf: symnmf.o
	$(CC) $(CFLAGS) -o symnmf symnmf.o -lm

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f symnmf symnmf.o

.PHONY: clean
