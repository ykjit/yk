// Run-time:
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_OPT=0
//   stderr:
//     6
//     yk-tracing: start-tracing
//     5
//     4
//     return
//     4
//     yk-tracing: stop-tracing
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
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}) ; @fprintf
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}) ; @fprintf
//     ...
//     --- End hir ---
//     3
//     yk-execution: enter-jit-code
//     2
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     --- End hir ---
//     1
//     return
//     exit

// Test that inlining into the main interpreter loop works when there is no
// closing / joining of traces in the "inner" frame. Note: the test output
// can't assert that there isn't a call to `loop` but by checking the number of
// `printf` calls, we implicitly check the property we care about.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *, YkLocation *, int, bool);

__attribute__((yk_unroll_safe))
void loop(YkMT *mt, YkLocation *loc1, YkLocation *loc2, int i, bool is_inner) {
  NOOPT_VAL(i);
  while (i > 0) {
    YkLocation *loc = loc1;
    if (i == 4)
      loc = loc2;
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    if (i == 5)
      loop(mt, loc1, loc2, i - 1, true);
    else if (i == 4 && is_inner)
      break;
    i--;
  }
  fprintf(stderr, "return\n");
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_null();

  loop(mt, &loc1, &loc2, 6, false);

  fprintf(stderr, "exit\n");
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
