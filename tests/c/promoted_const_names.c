// Run-time:
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     hello 0 4
//     ...
//     --- Begin hir ---
//     ...
//     %{{66}}: ptr = 0x{{_}} ; @__yk_opt_my_print0
//     call %{{66}}(%{{_}}) ; @__yk_opt_my_print0
//     ...
//     --- End hir ---
//     hello 0 3
//     hello 0 2
//     hello 0 1
//     exit

// Check that constant promoted values have their symbols attached, and that
// when a promoted function pointer resolves to a function that has an opt
// clone, the JIT calls the opt clone in the compiled trace.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

static void my_print0(int x) {
    fprintf(stderr, "hello 0 %d\n", x);
}

static void my_print1(int x) {
    fprintf(stderr, "hello 1 %d\n", x);
}

typedef void (*func_ty)(int);

void *funcptr_table[2] = {
    (void *)my_print0,
    (void *)my_print1,
};

static void call(int x, int id) {
    func_ty callable = (func_ty)yk_promote(funcptr_table[id]);
    callable(x);
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
    call(i, 0);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");

  return (EXIT_SUCCESS);
}
