// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   status: success

#include <assert.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int i = 10;
  YkMT *mt;
  YkLocation loc = yk_location_new();
  while (i > 0) {
    mt = yk_mt_new(NULL);
    yk_mt_hot_threshold_set(mt, 0);
    yk_mt_control_point(mt, &loc);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
