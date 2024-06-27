CPPFLAGS=-I/usr/include/ -I/usr/local/include/
CFLAGS=-fPIC -g -Wextra -Wpedantic

../../target/yk_gdb_plugin.so: yk_gdb_plugin.c
	${CC} ${CFLAGS} ${CPPFLAGS} -shared $< -o $@

.PHONY: clean
clean:
	-rm ../../target/yk_gdb_plugin.so
