// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     yk-tracing: start-tracing
//     we jumped
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected
//     ...

// Tests that we can deal with longjmp when we outline over it.

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

jmp_buf buf;

// This function will be outlined by the trace builder.
__attribute__((noinline))
void ljmp() {
  longjmp(buf, 1);
}

__attribute__((noinline, yk_outline))
void opaque() {
  ljmp();
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 2;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  if (setjmp(buf) != 0) {
    fprintf(stderr, "we jumped\n");
  }
  while (i > 0) {
    i--;
    yk_mt_control_point(mt, &loc);
    opaque();
    fprintf(stderr, "i=%d\n", i);
  }
  printf("exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
