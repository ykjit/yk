// ## yk-config-env: YKB_AOT_OPTLEVEL=1
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YK_LOG=4
//   env-var: YKD_LOG_STATS=/dev/null
//   stderr:
//     yk-jit-event: start-tracing
//     i=4, val=3
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{13_2}}: i32 = call f() ...
//     ...
//     %{{15_2}}: i32 = call g() ...
//     ...
//     %{{_}}: i32 = phi bb{{_}} -> %{{15_2}}, bb{{_}} -> %{{13_2}}
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{19}}: i32 = call @f()
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, %{{19}})
//     ...
//     --- End jit-pre-opt ---
//     i=3, val=3
//     yk-jit-event: enter-jit-code
//     i=2, val=3
//     i=1, val=3
//     yk-jit-event: deoptimise

// Check that PHI nodes JIT properly.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

bool test_compiled_event(YkCStats stats) {
  return stats.traces_compiled_ok == 1;
}

__attribute__((yk_outline)) int f() { return 3; }

__attribute__((yk_outline)) int g() { return 4; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (i == 3) {
      __ykstats_wait_until(mt, test_compiled_event);
    }
    int val;
    if (i > 0) {
      val = f();
    } else {
      val = g();
    }
    fprintf(stderr, "i=%d, val=%d\n", i, val);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
