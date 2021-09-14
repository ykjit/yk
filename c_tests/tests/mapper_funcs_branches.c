// Compiler:
// Run-time:

// Check that inter-procedural tracing works.

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk_testing.h>

__attribute__((noinline)) int add_one_or_two(int arg) {
  if (arg % 2 == 0) {
    arg++;
  } else {
    arg += 2;
  }
  return arg;
}

int main(int argc, char **argv) {
  __yktrace_start_tracing(HW_TRACING);
  argc = add_one_or_two(argc);
  void *tr = __yktrace_stop_tracing();

  size_t len = __yktrace_irtrace_len(tr);
  for (size_t i = 0; i < len; i++) {
    char *func_name = NULL;
    size_t bb = 0;
    __yktrace_irtrace_get(tr, i, &func_name, &bb);
    assert((strcmp(func_name, "main") == 0 ||
            strcmp(func_name, "add_one_or_two") == 0));
  }

  __yktrace_drop_irtrace(tr);
  return (EXIT_SUCCESS);
}
