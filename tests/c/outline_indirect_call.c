// ignore-if: test "$YK_JITC" = "j2"
// Run-time:
//   env-var: YKD_LOG_IR=jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     foo 7
//     yk-tracing: stop-tracing
//     --- Begin jit-post-opt ---
//     ...
//     %{{1}}: i32 = icall 0x...
//     ...
//     --- End jit-post-opt ---
//     foo 6
//     yk-execution: enter-jit-code
//     foo 5
//     foo 4
//     yk-execution: deoptimise ...
//     exit

// Check that an indirect call whose callee isn't marked `yk_indirect_inline`
// is outlined in the trace (even if the callee pointer is promoted).
//
// To have it inlined, the calle would need to be marked `yk_indirect_inline`.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int i) { return i + 3; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  int (*fn)(int) = foo;

  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(fn);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int (*fp)(int) = yk_promote((void *) fn);
    int x = fp(i);
    fprintf(stderr, "foo %d\n", x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
