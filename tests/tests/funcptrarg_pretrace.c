// ignore: broken during new control point design
// Compiler:
// Run-time:

// Test that indirect calls are only copied to the JITModule after we have seen
// `start_tracing`. Since indirect calls are handled before our regular
// are-we-tracing-yet check, and require an additional check, it makes sense to
// test for this here.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

int bar(size_t (*func)(const char *)) {
  int pre = func("abc");

  int res;
  __yktrace_start_tracing(HW_TRACING, &res, &func);
  res = func("abc");
  void *tr = __yktrace_stop_tracing();
  assert(res == 3);

  void *ptr = __yktrace_irtrace_compile(tr);
  __yktrace_drop_irtrace(tr);
  void (*cfunc)(void *, void *) = (void (*)(void *, void *))ptr;
  int output = 0;
  cfunc(&output, &func);
  assert(output == 3);

  assert(pre == 3);
  return res;
}

int main(int argc, char **argv) {
  int res = 0;
  res = bar(strlen);
  assert(res == 3);

  return (EXIT_SUCCESS);
}
