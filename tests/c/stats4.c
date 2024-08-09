// Run-time:
//   env-var: YKD_LOG_STATS=-
//   stderr:
//     {
//       ...
//       "traces_compiled_ok": ...
//       ...
//     }

// This tests that spun-up compile workers still cause stats to be printed out
// (i.e. that, because of reference counting, they don't keep `Mt` alive
// forever).

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  for (uint64_t i = 0; i < 2; i += 1) {
    yk_mt_control_point(mt, &loc);
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
