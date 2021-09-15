// ignore: impractical to test all optimisation levels.

// Check that debug information is included in module prints.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int res = 0;
  __yktrace_start_tracing(HW_TRACING, &res);
  res = 2;
  void *tr = __yktrace_stop_tracing();
  __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  return (EXIT_SUCCESS);
}
