// # Shadow stack doesn't currently support dynamically sized stack.
// ignore-if: true
// Run-time:
//   env-var: YKD_PRINT_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     define i8* @trace(...
//       ...
//       %{{a}} = add i64 %{{b}}, -1...
//       ...
//     }
//     ...
//     --- End aot ---
//     jit-state: enter-jit-code
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

#ifdef linux
#include <sys/sysinfo.h>
#endif

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
  YkLocation loc = yk_location_new();
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

#ifdef linux
  int n_thr = get_nprocs();
#else
#error unimplemented
#endif

  pthread_t threads[n_thr];
  struct thread_data tds[n_thr];
  for (int i = 0; i < n_thr; i++) {
    tds[i].tnum = i;
    tds[i].loc = &loc;
    tds[i].mt = mt;

    if (pthread_create(&threads[i], NULL, trace, &tds[i]) != 0)
      err(EXIT_FAILURE, "pthread_create");
  }

  void *thread_res = 0;
  for (int i = 0; i < n_thr; i++) {
    if (pthread_join(threads[i], &thread_res) != 0)
      err(EXIT_FAILURE, "pthread_join");
    assert((uintptr_t)thread_res == i);
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
