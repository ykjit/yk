// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stdout:
//     jihgfedcbajihgfedcbajihgfedcbajihgfedcbajihgfedcbajihgfedcbajihgfedcbajihgfedcbajihgfedcbajihgfedcba

// Check that guard failures in switches work as expected.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 3);
  YkLocation loc = yk_location_new();
  int i = 100;
  int j = 10;
  NOOPT_VAL(i);
  NOOPT_VAL(j);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    char c;
    switch (j % 10) {
    case 9:
      c = 'a';
      break;
    case 8:
      c = 'b';
      break;
    case 7:
      c = 'c';
      break;
    case 6:
      c = 'd';
      break;
    case 5:
      c = 'e';
      break;
    case 4:
      c = 'f';
      break;
    case 3:
      c = 'g';
      break;
    case 2:
      c = 'h';
      break;
    case 1:
      c = 'i';
      break;
    case 0:
      c = 'j';
      break;
    }
    printf("%c", c);
    i--;
    j++;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  printf("\n");

  return (EXIT_SUCCESS);
}
