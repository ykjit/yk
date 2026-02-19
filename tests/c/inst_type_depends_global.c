// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0 -Xclang -disable-O0-optnone -Xlinker --lto-newpm-passes=instcombine<max-iterations=1;no-verify-fixpoint>
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     %{{_}}: ptr = phi bb{{_}} -> @global1, bb{{_}} -> @global2
//     ...
//     --- End aot ---
//     ...

// Check that we can print and AOT IR instruction that requires asking the
// type of a global variable.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int global1 = 2000;
int global2 = 3000;

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(i);
    // makes a `phi` instruction whose type is determined by asking the type
    // of a global variable.
    int x = i % 2 == 0 ? global1 : global2;
    fprintf(stderr, "%i: %d\n", i, x);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
