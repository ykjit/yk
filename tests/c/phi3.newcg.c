// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_JITSTATE=-
//   env-var: YKD_LOG_STATS=/dev/null

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
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
