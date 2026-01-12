// ignore-if: test "$YK_JITC" != "j2" # formatting specific to j2
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: FUNC_ID=1
//   stdout:
//     hello 4
//     hello 3
//     hello 2
//     hello 1
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     ...; @my_print0
//     ...
//     --- End jit-pre-opt ---
//     ...

// Check that constant promoted values have their symbols attached.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

#define MAX_MSG 128

static void my_print0(int x) {
    printf("hello %d\n", x);
}

static void my_print1(int x) {
    printf("hello %d\n", x);
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

  const char *env_p = getenv("FUNC_ID");
  assert(env_p);
  int func_id = (int)(*env_p - '1');

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    call(i, func_id);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
