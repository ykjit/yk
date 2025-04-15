// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     func f(...
//     ...
//     %{{_}}: i{{size}} = icall %{{_}}(%{{_}})
//     ...
//     }
//     ...
//     --- End aot ---
//     ...

// Check that promotes in indirect callees of outlined functions are consumed
// properly during outlining. If we failed to consume them, an assertion would
// fail in the trace builder.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline))
uintptr_t g(uintptr_t x) {
  return yk_promote(x) + 1;
}

__attribute__((yk_outline))
uintptr_t f(uintptr_t x) {
  // force an indirect call.
  uintptr_t (*fptr)(uintptr_t) = g;
  NOOPT_VAL(fptr);
  return fptr(x);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%" PRIuPTR ": %" PRIuPTR "\n", i, f(1));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
