// ignore-if: test $YK_CARGO_PROFILE != "debug" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     # aot_debuginfo.c:{{_}}: YkMT *mt = yk_mt_new(NULL);
//     %{{_}}: ptr = call yk_mt_new(0x0)
//     ...
//     # aot_debuginfo.c:{{_}}: yk_mt_hot_threshold_set(mt, 0);
//     call yk_mt_hot_threshold_set(%{{_}}, 0i32)
//     ...
//     # aot_debuginfo.c:{{_}}: YkLocation loc = yk_location_new();
//     ...
//     %{{_}}: i64 = call yk_location_new()
//     ...
//     # aot_debuginfo.c:{{_}}: int i = 4;
//     *%{{_}} = 4i32
//     ...
//     # aot_debuginfo.c:{{_}}: while (i > 0) {
//     ...
//     condbr %{{_}}, bb{{_}}, bb{{_}}...
//     ...
//     # aot_debuginfo.c:{{_}}: yk_mt_control_point(mt, &loc);
//     ...
//     call __ykrt_control_point(%{{_}}, %{{_}}, 1i64) [safepoint:...
//     ...
//     # aot_debuginfo.c:{{_}}: i--;
//     ...
//     %{{_}}: i32 = add %{{_}}, -1i32
//     ...
//     # aot_debuginfo.c:{{_}}: yk_location_drop(loc);
//     ...
//     call yk_location_drop(%{{_}})
//     ...
//     # aot_debuginfo.c:{{_}}: yk_mt_shutdown(mt);
//     ...
//     call yk_mt_shutdown(%{{_}})
//     ...
//     # aot_debuginfo.c:{{_}}: return (EXIT_SUCCESS);
//     ...

// Check that debug information is included when the AOT module prints.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
