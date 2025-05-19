// ## FIXME: Implement setjmp/longjmp detection for swt.
// ignore-if: test "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     yk-tracing: start-tracing
//     we jumped
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: longjmp encountered
//     ...

// Tests that we can deal with setjmp/longjmp.

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

jmp_buf buf;

void ljmp() {
  int r = 0;
  for (int i = 0; i < 10; i++) {
    r += 1;
  }
  longjmp(buf, r);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  int r = setjmp(buf);
  if (r == 10) {
    fprintf(stderr, "we jumped\n");
  }
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (r != 10) {
      ljmp();
    }
    fprintf(stderr, "i=%d\n", i);
    res += 2;
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
