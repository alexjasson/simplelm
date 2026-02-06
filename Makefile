CC = gcc
CFLAGS = -Wall -Wextra -g
SRC = src

chat: src/chat.c src/Model.c src/Model.h src/Matrix.c src/Matrix.h src/utility.c src/utility.h
	$(CC) $(CFLAGS) -o chat src/chat.c src/Model.c src/Matrix.c src/utility.c -lm

clean:
	rm -f chat

.PHONY: clean
