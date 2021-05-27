// Compiler:
// Run-time:

// Check that trace compilation works in the non-entry block.
//
// Since LLVM allocas typically appear in the entry block of a function, we
// will miss the allocas if tracing starts in a later block.
//
// FIXME An optimising compiler can remove all of the code between start/stop
// tracing.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  // Causes the traced block to NOT be the entry block.
  if (argc == -1)
    abort();

  int res;
  void *tt = __yktrace_start_tracing(HW_TRACING);
  // Causes both a load and a store to things defined outside the trace.
  res = 1 + argc;
  void *tr = __yktrace_stop_tracing(tt);

  assert(res == 2);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*func)() = (void (*)())ptr;
  func();

  return (EXIT_SUCCESS);
}
