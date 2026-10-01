// ignore-if: test ${YK_ARCH} != "x86_64"
// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   stderr:
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
//     --- End hir ---
//     --- Begin hir ---
//     ; {
//     ;   "trid": "1",
//     ;   "start": {
//     ;     "kind": "Guard",
//     ;     "src_trid": "0",
//     ;     "gidx": "{{_}}"
//     ;   },
//     ;   "end": {
//     ;     "kind": "Return"
//     ;   }
//     ; }
//     ...
//     --- End hir ---
//     4 4 4

// Check that callee saved registers really are saved, both for deopt and return traces.

#include <assert.h>
#include <stdio.h>
#include <yk.h>

extern int call_check_csrs(YkMT *, YkLocation *, int);

__attribute__((noinline))
int interp(YkMT *mt, YkLocation *loc, int i) {
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    if (i == 4)
      return i;
    --i;
  }
  return i;
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  YkLocation loc = yk_location_new();
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 1);
  int x = call_check_csrs(mt, &loc, 10);
  int y = call_check_csrs(mt, &loc, 10);
  int z = call_check_csrs(mt, &loc, 10);
  fprintf(stderr, "%d %d %d\n", x, y, z);
  yk_mt_shutdown(mt);
  yk_location_drop(loc);
  return 0;
}
