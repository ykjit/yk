// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 0x...
//     yk-tracing: stop-tracing
//     3: 0x...
//     yk-execution: enter-jit-code
//     2: 0x...
//     1: 0x...
//     yk-execution: deoptimise
//     exit

// Check that outlining over a promoted pointer works.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>


__attribute__((noinline))
void *g(void *p) {
  p = yk_promote(p);
  return p;
}

__attribute__((yk_outline))
void *f(void *p) {
  return g(p);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %p\n", i, f(&i));
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
