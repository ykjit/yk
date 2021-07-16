// Compiler:
// Run-time:

// Check that inlining a function with a void return type works.
//
// FIXME An optimising compiler can remove all of the code between start/stop
// tracing.

#include <assert.h>
#include <stdlib.h>
#include <yk_testing.h>

void __attribute__((noinline)) f() { return; }

int main(int argc, char **argv) {
  void *tt = __yktrace_start_tracing(HW_TRACING);
  f();
  void *tr = __yktrace_stop_tracing(tt);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)() = (void (*)())ptr;
  func();

  return (EXIT_SUCCESS);
}
