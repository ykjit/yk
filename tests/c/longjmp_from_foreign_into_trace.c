// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     yk-warning: trace-compilation-aborted: irregular control flow detected
//     ...

// Tests that we can deal with setjmp/longjmp when we jump from foreign code
// into a trace.

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

jmp_buf buf;

__attribute__((noinline, yk_outline))
void ljmp() {
  longjmp(buf, 1);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 2;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (i == 2) {
      if (setjmp(buf) == 0) {
        fprintf(stderr, "setjmp returned 0\n");
        ljmp();
      } else {
        fprintf(stderr, "setjmp returned non-zero\n");
      }
    }
    fprintf(stderr, "i=%d\n", i);
    i--;
  }
  printf("exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
