// ignore-if: test "$YK_JITC" = "j2" # not yet implemented in j2
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     debug_str %{{10_0}}
//     ...
//     ...call fprintf(...
//     ...
//     debug_str %{{13_0}}
//     ...
//     --- End aot ---
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     ; debug_str: before fprintf: 4
//     ...
//     ...call @fprintf(...
//     ...
//     ; debug_str: after fprintf: 5
//     ...
//     --- End jit-pre-opt ---
//     ...
//     --- Begin jit-post-opt ---
//     ...
//     ; debug_str: before fprintf: 4
//     ...
//     ...call @fprintf(...
//     ...
//     ; debug_str: after fprintf: 5
//     ...
//     --- End jit-post-opt ---
//     ...

// Check that yk_debug_str() works and is not optimised out.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

#define MAX_MSG 128

__attribute__((noinline))
void g() {
  yk_debug_str("inside g");
}

__attribute__((yk_outline))
void f() {
  g();
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  char msg[MAX_MSG];

  // Prevent the OutlineUntraceable pass from marking g() yk_outline.
  g();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    snprintf(msg, MAX_MSG, "before fprintf: %d", i);
    yk_debug_str(msg);
    f();
    fprintf(stderr, "%d\n", i);
    snprintf(msg, MAX_MSG, "after fprintf: %d", i + 1);
    yk_debug_str(msg);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
