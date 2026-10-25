// Run-time:
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     --- Begin hir ---
//     ...
//     %{{_}}: ptr = threadlocal tlsv
//     ...
//     --- End hir ---
//     yk-execution: enter-jit-code {"trid": "0"}
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     tlsv: 10
//     yk-execution: enter-jit-code {"trid": "0"}
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     tlsv: 11
//     tlsv: 10

// Check llvm.threadlocal.address works.

#include <assert.h>
#include <err.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__thread volatile int tlsv = 0;

__attribute__((noinline)) void add_tls(int value) { tlsv += value; }

struct thread_data {
  YkLocation *loc;
  YkMT *mt;
  int initial_val;
};

static void *interp(void *arg) {
  struct thread_data *td = (struct thread_data *)arg;
  assert(tlsv == 0);
  tlsv = td->initial_val;
  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(td->mt, td->loc);
    add_tls(i);
    i--;
  }
  fprintf(stderr, "tlsv: %d\n", tlsv);
  return NULL;
}

int main() {
  YkLocation loc = yk_location_new();
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  struct thread_data data = {&loc, mt, 0};

  interp(&data);
  pthread_t thread;
  data.initial_val = 1;
  if (pthread_create(&thread, NULL, interp, &data) != 0)
    err(EXIT_FAILURE, "pthread_create");
  if (pthread_join(thread, NULL) != 0)
    err(EXIT_FAILURE, "pthread_join");
  fprintf(stderr, "tlsv: %d\n", tlsv);

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
