// Run-time:
//   stdout: 205

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

// external function
int call_callback(int (*callback)(int, int), int x, int y);

// callbacks
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

int main(int argc, char **argv) {
  int i = 0;
  YkMT *mt = yk_mt_new(NULL);
  YkLocation loc = yk_location_new();
  int result = 0;
  while (i < 10) {
    yk_mt_control_point(mt, &loc);
    if (i % 2 == 0) {
      result += call_callback(add, i, i);
    } else {
      result += call_callback(mul, i, i);
    }
    i++;
  }
  printf("%d", result);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
