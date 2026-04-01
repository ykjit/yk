// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_LOG=4
//   stderr:
//     7
//     yk-tracing: start-tracing
//     6
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     --- End hir ---
//     5
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     early return
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     ; {
//     ;   "trid": "{{_}}",
//     ;   "start": {
//     ;     "kind": "Guard",
//     ;     "src_trid": "{{_}}",
//     ;     "gidx": "{{_}}"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Return"
//     ;   }
//     ; }
//     ...
//     term [%{{_}}]
//     --- End hir ---
//     3
//     yk-execution: enter-jit-code
//     2
//     1
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     return
//     4 0
//     exit

// Check that (Guard, Return) traces work correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int loop(YkMT *, YkLocation *, int);

int loop(YkMT *mt, YkLocation *loc, int i) {
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    if (i == 4) {
      fprintf(stderr, "early return\n");
      return i;
    }
    fprintf(stderr, "%d\n", i);
    i--;
  }
  fprintf(stderr, "return\n");
  return i;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  NOOPT_VAL(loc);
  int x = loop(mt, &loc, 7);
  int y = loop(mt, &loc, 3);
  fprintf(stderr, "%d %d\n", x, y);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
