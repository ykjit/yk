// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     0: 5
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ; {
//     ;   "trid": "0",
//     ;   "start": {
//     ;     "kind": "ControlPoint"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Call"
//     ;   }
//     ; }
//     ...
//     %{{16}}: i1 = 0
//     ...
//     %{{24}}: ptr = 0x{{_}} ; @interp
//     %{{25}}: i32 = call %{{23}}(%{{_}}, %{{_}}, %{{_}}, %{{_}}) ; @interp
//     guard true, %{{16}}, [%{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{25}}]
//     term []
//     --- End hir ---
//     1: 4
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 3
//     yk-execution: enter-jit-code {"trid": "0"}
//     3: 2
//     yk-execution: deoptimise {"trid": "0", "gidx": "1"}
//     yk-tracing: start-side-tracing
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ; {
//     ;   "trid": "1",
//     ;   "start": {
//     ;     "kind": "Guard",
//     ;     "src_trid": "0",
//     ;     "gidx": "1"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Coupler",
//     ;     "tgt_trid": "0"
//     ;   }
//     ; }
//     ...
//     --- End hir ---
//     3: 1
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}
//     yk-tracing: start-side-tracing
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ; {
//     ;   "trid": "2",
//     ;   "start": {
//     ;     "kind": "Guard",
//     ;     "src_trid": "0",
//     ;     "gidx": "0"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Coupler",
//     ;     "tgt_trid": "0"
//     ;   }
//     ; }
//     ...
//     --- End hir ---
//     2: 2
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 1
//     yk-execution: deoptimise {"trid": "1", "gidx": "0"}
//     yk-tracing: start-side-tracing
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ; {
//     ;   "trid": "3",
//     ;   "start": {
//     ;     "kind": "Guard",
//     ;     "src_trid": "1",
//     ;     "gidx": "0"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Return"
//     ;   }
//     ; }
//     ...
//     %{{9}}: i32 = add ...
//     ...
//     term [%{{9}}]
//     --- End hir ---
//     1: 3
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 2
//     2: 1
//     yk-execution: return {"trid": "0"}
//     yk-execution: enter-jit-code {"trid": "0"}
//     1: 2
//     1: 1
//     yk-execution: return {"trid": "0"}
//     yk-execution: enter-jit-code {"trid": "0"}
//     0: 4
//     yk-execution: enter-jit-code {"trid": "0"}
//     1: 3
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 2
//     2: 1
//     yk-execution: return {"trid": "0"}
//     1: 2
//     1: 1
//     yk-execution: return {"trid": "0"}
//     0: 3
//     yk-execution: enter-jit-code {"trid": "0"}
//     1: 2
//     1: 1
//     yk-execution: return {"trid": "0"}
//     0: 2
//     0: 1
//     yk-execution: return {"trid": "0"}
//     rtn: 6
//     exit

// Test the call traces work with functions that return values.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int interp(YkMT *mt, YkLocation *loc, int recurse, int i){
  int rtn;
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d: %d\n", recurse, i);
    if (i > 2)
      rtn = interp(mt, loc, recurse + 1, i - 1);
    i--;
  }
  return rtn + recurse;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int rtn = interp(mt, &loc, 0, 5);
  fprintf(stderr, "rtn: %d\n", rtn);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return (EXIT_SUCCESS);
}
