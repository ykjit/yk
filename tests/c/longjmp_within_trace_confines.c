// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     i=3
//     after setjmp
//     after setjmp
//     yk-tracing: start-tracing
//     i=2
//     after setjmp
//     after setjmp
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected
//     i=1
//     after setjmp
//     after setjmp
//     exit

// Check that something sensible happens when there's a longjmp within the
// confines of the trace.

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();
  jmp_buf buf;

  int i = 3;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);

    int r = setjmp(buf);
    fprintf(stderr, "after setjmp\n");

    if (r == 0) {
      longjmp(buf, 1);
      fprintf(stderr, "we jumped\n");
    }
    i--;
  }
  fprintf(stderr, "exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
