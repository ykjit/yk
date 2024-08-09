// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     ashr 4
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{result}}: i64 = ashr %{{1}}, 2i64
//     ...
//     --- End jit-pre-opt ---
//     ashr 3
//     yk-jit-event: enter-jit-code
//     ashr 2
//     ashr 1
//     yk-jit-event: deoptimise
//     exit

// Test ashr instructions with the exact keyword.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int i) { return i + 3; }

struct P {
  int *yklocs;
  int *code;
};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  struct P p;
  p.yklocs = malloc(sizeof(int) * 4);
  p.code = &p.yklocs[0];
  p.yklocs[4] = 4;
  p.yklocs[3] = 3;
  p.yklocs[2] = 2;
  p.yklocs[1] = 1;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int *elem = &p.yklocs[i];
    int ashr = p.yklocs[elem - p.code];
    fprintf(stderr, "ashr %d\n", ashr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
