// Run-time:
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=aot,hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-tracing: start-tracing
//     ...
//     yk-tracing: stop-tracing
//     ...
//     call llvm.experimental.patchpoint.void(0i64, 13i32, __ykrt_control_point, 3i32, %7_0, @location, 0i64) [statepoint: 0i64, ()]
//     ...
//     --- Begin hir ---
//     ; {
//     ...
//     ; }
//     %0: ptr = 0x{{_}} ; @global_var
//     %1: {{_}} = load %0
//     ...
//     %0: ptr = 0x{{_}} ; @global_var
//     %1: {{_}} = load %0
//     ...
//     --- End hir ---
//     ...
//     yk-execution: enter-jit-code
//     ...

// Testing that when all interpreter state is in globals the JIT produces
// a trace block with no Arg instructions.
//
// The AOT patchpoint passes `@location` and `@global_var` as global address
// constants (not live locals). The empty statepoint `()` confirms this: live
// locals would appear there if interpreter state were in local variables.



#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

YkMT *g_mt = NULL;
YkLocation location;
long global_var = 0;

int main(void) {
  g_mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(g_mt, 0);
  yk_mt_sidetrace_threshold_set(g_mt, 4);
  location = yk_location_new();

  for (; global_var < 20; global_var++) {
    yk_mt_control_point(g_mt, &location);
  }

  yk_location_drop(location);
  yk_mt_shutdown(g_mt);
  return EXIT_SUCCESS;
}
