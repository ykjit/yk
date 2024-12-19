// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     11
//     10
//     9
//     longjmp
//     8
//     7
//     6
//     longjmp
//     5
//     4
//     3
//     longjmp
//     2
//     1
//     0
//     longjmp

// Check that the ykllvm shadow stack pass works at runtime in the presence of
// setjmp and longjump.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>
#include <setjmp.h>

__attribute__((noinline))
void g(int i, jmp_buf *env) {
  fprintf(stderr, "%d\n", i);
  if (i % 3 == 0) {
    fprintf(stderr, "longjmp\n");
    longjmp(*env, 1);
  }
}

__attribute__((noinline))
void f(int i, jmp_buf *env) {
  while (i >= 0) {
    g(i, env);
    i--;
  }
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  jmp_buf env;

  int i = 11;
  NOOPT_VAL(i);
  while (i > 0) {
    // Passing a NULL location. We never JIT. Just checking AOT behaviour.
    yk_mt_control_point(mt, NULL);
    if (setjmp(env) != 0) {
      i -= 3;
    }
    f(i, &env);
    i--;
  }
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
