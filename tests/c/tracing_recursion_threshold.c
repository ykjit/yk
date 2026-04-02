// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_OPT=0
//   stderr:
//     7
//     inner 6
//     inner 5
//     inner 4
//     inner 3
//     inner 2
//     inner 1
//     return
//     --- Begin hir ---
//     ; {
//     ;   "trid": "0",
//     ;   "start": {
//     ;     "kind": "ControlPoint"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Loop"
//     ;   }
//     ; }
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}) ; @fprintf
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}) ; @fprintf
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}) ; @fprintf
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}) ; @fprintf
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}) ; @fprintf
//     ...
//     call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{_}}) ; @loop
//     ...
//     --- End hir ---
//     6
//     inner 5
//     inner 4
//     inner 3
//     inner 2
//     inner 1
//     return
//     5
//     inner 4
//     inner 3
//     inner 2
//     inner 1
//     return
//     4
//     inner 3
//     inner 2
//     inner 1
//     return
//     3
//     inner 2
//     inner 1
//     return
//     2
//     inner 1
//     return
//     1
//     return
//     return
//     exit


// Test that we inline up to a given threshold.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *, int, bool);

__attribute__((yk_unroll_safe))
void loop(YkMT *mt, YkLocation *loc, int i, bool is_inner) {
  if (is_inner && i > 0) {
    fprintf(stderr, "inner %d\n", i);
    loop(mt, loc, i - 1, true);
    return;
  }
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    loop(mt, loc, i - 1, true);
    i--;
  }
  fprintf(stderr, "return\n");
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  loop(mt, &loc, 7, false);

  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
