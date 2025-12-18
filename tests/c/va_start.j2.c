// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 50
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     call llvm.va_start(%{{_}})
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}) ; @__yk_opt_f
//     ...
//     --- End jit-pre-opt ---
//     3: 36
//     yk-execution: enter-jit-code
//     2: 23
//     1: 11
//     yk-execution: deoptimise ...
//     exit

// Check that functions using `va_start` and `va_end` (etc.) are handled
// correctly.
//
// Ideally this would mean inlining them, but this has proven difficult, since
// `va_*` are not really functions, but compiler specific macros. For example,
// at the time of writing `va_start` is macro'd to the `__builtin_va_start`
// builtin (see `__stdarg_va_arg.h`) which is specially recognised in LLVM's
// MIR pipeline and a series of machine blocks are directly inlined.
//
// In light of this, this test currently checks that functions containing
// `va_start` are outlined, even if they are marked `yk_outline_safe`.

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

// Sums the `n` vararg integers.
__attribute__((noinline,yk_unroll_safe))
int f(int n, ...) {
  va_list ap;
  va_start(ap, n);
  int sum = 0;
  while (n > 0) {
    sum += va_arg(ap, int);
    n--;
  }
  va_end(ap);
  return sum;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %d\n", i, f(i, 11, 12, 13, 14));
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
