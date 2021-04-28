all: blockmap
	env LD_LIBRARY_PATH=${SYSROOT_LIBDIR} ./blockmap

blockmap:
	clang ${CFLAGS} ${LDFLAGS} -o blockmap blockmap.c
