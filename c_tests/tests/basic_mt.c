// Compiler:
// Run-time:

#include <yk.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt();
  YkLocation loc = yk_location_new();
  for (int i = 0; i < yk_mt_hot_threshold(mt); i++) {
    yk_control_point(mt, &loc);
  }
  yk_location_drop(loc);
  return 0;
}
