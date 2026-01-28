CC = gcc
CFLAGS = -fsanitize=undefined -Wall -Wextra -pedantic

chat:
	$(CC) $(CFLAGS) -o chat chat.c Model.c -lm

.PHONY: clean

clean:
	rm -f *.o chat
