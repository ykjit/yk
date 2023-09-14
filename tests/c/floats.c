// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   status: success

#include <stdio.h>
#include <yk.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  float i = 0;
  YkLocation loc = yk_location_new();
  while (i < 3) {
    yk_mt_control_point(mt, &loc);
    i += 1.0;
  }
  return 0;
}
