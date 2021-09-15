// Compiler:
// Run-time:

// Check that compiling and running multiple traces in sequence works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

void trace(void) {
  __yktrace_start_tracing(HW_TRACING);
  int res = 1 + 1;
  void *tr = __yktrace_stop_tracing();
  assert(res == 2);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)() = (void (*)())ptr;
  func();
}

int main(int argc, char **argv) {
  for (int i = 0; i < 3; i++)
    trace();

  return (EXIT_SUCCESS);
}
