CC = gcc
CFLAGS = -Wall -Wextra -g
SRC = src

chat: src/chat.c src/Model.c src/Model.h
	$(CC) $(CFLAGS) -o chat src/chat.c src/Model.c -lm

clean:
	rm -f chat

.PHONY: clean
