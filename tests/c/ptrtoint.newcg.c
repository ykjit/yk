// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     ptr: {{ptr}}
//     jitstate: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i64 = zext %{{2}}, i64
//     ...
//     --- End jit-pre-opt ---
//     ptr: {{ptr}}
//     jitstate: enter-jit-code
//     ptr: {{ptr}}
//     ptr: {{ptr}}
//     jitstate: deoptimise
//     exit

// Check that basic trace compilation works.

// FIXME: Get this test all the way through the new codegen pipeline!
//
// Currently it succeeds even though it crashes on deopt. This is so
// that we can incrementally implement the new codegen and have CI merge our
// incomplete work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    long ptr = (long)&loc;
    fprintf(stderr, "ptr: %ld\n", ptr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
