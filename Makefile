# Detect compiler
ifneq ($(shell which gcc 2>/dev/null),)
    CC := gcc
else
    $(error Compiler not found! Please install gcc)
endif

CFLAGS = -Wall -Wextra -Werror -pedantic -Ofast -march=native -flto

chat: src/chat.c src/Model.c src/Model.h src/Matrix.c src/Matrix.h src/utility.c src/utility.h src/ThreadPool.c src/ThreadPool.h
	$(CC) $(CFLAGS) -o chat src/chat.c src/Model.c src/Matrix.c src/utility.c src/ThreadPool.c -lm -lpthread

clean:
	rm -f chat

.PHONY: clean
