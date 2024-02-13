// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_STATS=-
//   stderr:
//     {
//       ...
//       "trace_executions": 0,
//       "traces_compiled_err": 1,
//       "traces_compiled_ok": 0,
//       "traces_recorded_err": 0,
//       "traces_recorded_ok": 1
//       ...
//     }

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  for (int i = 0; i < 2; i += 1) {
    yk_mt_control_point(mt, &loc);
    jmp_buf env;
    setjmp(env);
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
