// ignore-if: test "$YKB_TRACER" != "swt"
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-post-opt,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     func add(%arg0: i32, %arg1: i32) -> i32 {
//     ...
//     func dec(%arg0: i32) -> i32 {
//     ...
//     %{{_}}: i32 = call add(%{{_}}, -1i32) [safepoint: 2i64, (%{{_}}, %{{_}})]
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     *%{{_}} = dec
//     ...
//     %{{_}}: i32 = icall %{{_}}(%{{_}})
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i32 = icall %{{_}}(%{{_}})
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     %{{_}}: i32 = icall %{{_}}(%{{_}})
//     ...
//     --- End jit-post-opt ---
//     3
//     yk-jit-event: enter-jit-code
//     2
//     1
//     yk-jit-event: deoptimise
//     exit
//   status: success

// Check that functions which address is taken can refer to other
// functions which address is not taken. This test is specific for SWT
// with module cloning enabled. Note that the cloned functions will not
// be visible in the aot ir since they are not serialised.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_outline)) int add(int i, int j) { return i + j; }

__attribute__((yk_outline)) int dec(int i) { return add(i, -1); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);

  // Take a reference to the 'dec' function using a function pointer.
  // This will cause dec function to not be cloned.
  int (*dec_ptr)(int) = dec;

  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d\n", i);
    i = dec_ptr(i);
  }
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
