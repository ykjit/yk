// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YK_LOG=4
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     ...

// Test that indirect calls are only copied to the JITModule after we have seen
// `start_tracing`. Since indirect calls are handled before our regular
// are-we-tracing-yet check, and require an additional check, it makes sense to
// test for this here.
//
// FIXME: since lang_tester cannot do negative matching, we can't check for the
// absence of the indirect call. For now we are only checking that nothing
// crashes.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int bar(size_t (*func)(const char *)) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t pre = func("abc");
  int i = 2;
  NOOPT_VAL(pre);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return pre;
}

int main(int argc, char **argv) {
  int res = 0;
  res = bar(strlen);
  assert(res == 3);

  return (EXIT_SUCCESS);
}
