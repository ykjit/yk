// ## This tests that traces that generate too many blocks cause "trace too
// ## long" warnings. This can be very slow (e.g. on swt), so ignore it except
// ## where we know it'll run fast enough.
// ##
// ## FIXME: doesn't trigger "trace too long".
// ignore-if: test "$YKB_TRACER" != "swt"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     ...
//     yk-warning: stop-tracing-aborted: Trace too long

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();

  int i = 2000;
  NOOPT_VAL(loc1);
  NOOPT_VAL(loc2);
  NOOPT_VAL(i);
  YkLocation *loc = &loc1;
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    if (i == 2000)
      loc = &loc2;
    else if (i == 2)
      loc = &loc1;
    fprintf(stdout, "i=%d\n", i);
    i--;
  }
  NOOPT_VAL(i);
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
