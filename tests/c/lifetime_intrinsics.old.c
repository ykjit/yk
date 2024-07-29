// Run-time:
//   env-var: YKD_LOG_IR=-:aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     define dso_local i32 @main(...
//       ...
//       call void @llvm.lifetime.start...
//       ...
//     }
//     ...
//     --- End aot ---
//     ...

// Check that we can handle llvm.lifetime.start/stop.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    // Variable definition in an inner scope causes lifetime annotations.
    int j = 5;
    NOOPT_VAL(j);
    fprintf(stderr, "%d:%d\n", i, j);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
