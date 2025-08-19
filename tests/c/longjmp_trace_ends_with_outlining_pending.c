// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     enter
//     in loop: i=4
//     yk-tracing: start-tracing
//     enter
//     in loop: i=4
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected (trace ended with outline successor pending)
//     jumped
//     in loop: i=3
//     yk-tracing: start-tracing
//     jumped
//     in loop: i=2
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: irregular control flow detected (unexpected successor)
//     jumped
//     in loop: i=1
//     yk-tracing: start-tracing
//     jumped
//     exit

// Check that we protect against longjmps that cause the trace builder to
// remain in outlining mode when the trace ends.

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

jmp_buf buf;
bool flag = false;


void loop(YkMT *mt, YkLocation *loc) {
  fprintf(stderr, "enter\n");
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    fprintf(stderr, "in loop: i=%d\n", i);
    yk_mt_control_point(mt, loc);
    if (!flag) {
      if (setjmp(buf) == 0) {
        flag = true;
        loop(mt, loc);
      } else {
        fprintf(stderr, "jumped\n");
      }
    } else {
      longjmp(buf, 1);
    }
    i--;
  }
}

__attribute__((noinline, yk_outline))
int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  loop(mt, &loc);
  fprintf(stderr, "exit");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
