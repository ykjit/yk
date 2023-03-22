#include <setjmp.h>
#include <stdbool.h>
#include <stdio.h>

#include "yk_testing.h"

int call_me(int x) { return 5; }

int call_me_add(int x) { return x + 1; }

int call_callback(int (*callback)(int, int), int x, int y) {
  return callback(x, y);
}

jmp_buf jbuf;
void unmapped_setjmp(void) {
  if (setjmp(jbuf) == 0) {
    fprintf(stderr, "set jump point\n");
    NOOPT_VAL(jbuf);
    longjmp(jbuf, 1);
  } else {
    fprintf(stderr, "jumped!\n");
  }
}
