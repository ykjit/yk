// ## Shadow stack isn't thread safe.
// ignore-if: true
// Run-time:
//   env-var: YK_LOG=4
//   stderr:
//     ...
//     yk-jit-event: enter-jit-code
//     ...

// Check that compiling and running traces in parallel works.

#include <assert.h>
#include <err.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

#define NUM_THREADS 8
#define ITERS 100000

struct thread_data {
  int tnum; // The thread's "number".
  YkLocation *loc;
  YkMT *mt;
};

// Decrement an integer from ITERS down to the thread's number, then return it.
static void *trace(void *arg) {
  struct thread_data *td = (struct thread_data *)arg;

  uintptr_t i = ITERS;
  NOOPT_VAL(i);
  while (i != td->tnum) {
    yk_mt_control_point(td->mt, td->loc);
    i--;
  }
  NOOPT_VAL(i);
  return (void *)i;
}

int main() {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  pthread_t threads[NUM_THREADS];
  struct thread_data tds[NUM_THREADS];
  YkLocation locs[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    locs[i] = yk_location_new();
    tds[i].tnum = i;
    tds[i].loc = &locs[i];
    tds[i].mt = mt;

    if (pthread_create(&threads[i], NULL, trace, &tds[i]) != 0)
      err(EXIT_FAILURE, "pthread_create");
  }

  void *thread_res = 0;
  for (int i = 0; i < NUM_THREADS; i++) {
    if (pthread_join(threads[i], &thread_res) != 0)
      err(EXIT_FAILURE, "pthread_join");
    assert((uintptr_t)thread_res == i);
    yk_location_drop(locs[i]);
  }

  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
