// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt,aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     define dso_local void @g(...
//     ...
//     define dso_local void @f(...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     call void @f(...
//     ...
//     --- End jit-pre-opt ---
//     ...
//     jitstate: enter-jit-code
//     f: 3
//     g: 3
//     f: 2
//     g: 2
//     f: 1
//     g: 1
//     f: 0
//     ...

// Check that mutual recusrion works.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

void f(int);
void g(int);

__attribute__((noinline)) void g(int num) {
  NOOPT_VAL(num);
  fprintf(stderr, "g: %d\n", num);
  num--;
  f(num);
}

__attribute__((noinline)) void f(int num) {
  NOOPT_VAL(num);
  fprintf(stderr, "f: %d\n", num);
  if (num == 0)
    return;

  g(num);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = 3;
    NOOPT_VAL(x);
    f(x);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
