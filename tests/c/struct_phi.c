
// Check that we can handle struct field accesses where the field is
// initialised via a phi node.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

struct s {
  int x;
};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int z = 5;
  struct s s1;
  s1.x = argc || z; // Creates a phi node at -O0.
  int y1 = 0, i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(s1);
    y1 = s1.x;
    NOOPT_VAL(y1);
    i--;
  }
  assert(y1 == 1);

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
