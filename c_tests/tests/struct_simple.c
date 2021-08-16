// ignore: https://github.com/ykjit/yk/issues/409
// Compiler:
// Run-time:

// Check that we can handle struct field accesses.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

struct s {
  int x;
};

int main(int argc, char **argv) {
  struct s s1 = {argc};
  int y1 = 0;
  void *tt = __yktrace_start_tracing(HW_TRACING, &y1, &s1);
  NOOPT_VAL(s1);
  y1 = s1.x;
  NOOPT_VAL(y1);
  void *tr = __yktrace_stop_tracing(tt);
  assert(y1 == 1);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)(int *, struct s *) = (void (*)(int *, struct s *))ptr;
  int y2 = 0;
  func(&y2, &s1);
  printf("%d\n", y2);
  assert(y2 == 1);

  return (EXIT_SUCCESS);
}
