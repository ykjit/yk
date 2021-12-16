TARGET_DIR = $(shell readlink -f $(shell pwd)/../../target/)

simple.o:
	clang simple.ll -o ${TARGET_DIR}/simple.o

deopt.o:
	clang deopt.ll -o ${TARGET_DIR}/deopt.o

simple.ll:
	clang -emit-llvm -S -o simple.ll -c simple.c

deopt.ll:
	clang -emit-llvm -S -o deopt.ll -c deopt.c

clean:
	rm simple.o
	rm deopt.o
