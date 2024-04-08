// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   status: error
//   stderr:
//     jitstate: start-tracing
//     foo
//     jitstate: stop-tracing
//     --- Begin aot ---
//     ...
//     func main($arg0: i32, $arg1: ptr) -> i32 {
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i8 = Icmp %{{2}}, SignedGreater, 1i32
//     ...
//     %{{3}}: i64 = Call @fwrite(%{{4}}, 4i64, 1i64, %{{5}})
//     ...
//     --- End jit-pre-opt ---
//     foo
//     jitstate: enter-jit-code
//     foo
//     jitstate: deoptimise
//     bar

// Check that call inlining works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void foo(int i) {
  if (i > 1) {
    fputs("foo\n", stderr);
  } else {
    fputs("bar\n", stderr);
  }
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
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    foo(i);
    res += 2;
    i--;
  }
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
