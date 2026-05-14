// Run-time:
//   stderr:
//     ...onshadow=1
//     ...onshadow=1
//     ...onshadow=1
//     ...onshadow=1
//     ...onshadow=1
//     ...onshadow=1
//     ...onshadow=1
//     ...onshadow=1

// Check we can get the shadow stack bounds for the current thread.
// The control point in `f` allocates a shadow stack for each thread.

#include <assert.h>
#include <err.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define NUM_THREADS 8

void *f(void *arg) {
  YkMT *mt = (YkMT *)arg;
  YkLocation loc = yk_location_new();
  // yk_mt_control_point requires to be in a loop.
  int x = 0;
  while(x == 0) {
    NOOPT_VAL(x);
    // Control point allocates the shadow stack for this thread.
    yk_mt_control_point(mt, &loc);
    void *ss_start = NULL, *ss_end = NULL;
    yk_thread_shadowstack_bounds(&ss_start, &ss_end);
    int inbounds = ((void *)&x >= ss_start) && ((void *)&x <= ss_end);
    fprintf(stderr, "&x=%p, ss=[%p, %p], onshadow=%d\n", &x, ss_start, ss_end,
            inbounds);
    assert(ss_start < ss_end);
    assert(inbounds);
    x = 1;
  }
  return NULL;
}

int main() {
  YkMT *mt = yk_mt_new(NULL);
  pthread_t threads[NUM_THREADS];
  for (int j = 0; j < NUM_THREADS; j++) {
    if (pthread_create(&threads[j], NULL, f, mt) != 0)
      err(EXIT_FAILURE, "pthread_create");
  }

  void *thread_res = NULL;
  for (int j = 0; j < NUM_THREADS; j++) {
    if (pthread_join(threads[j], &thread_res) != 0)
      err(EXIT_FAILURE, "pthread_join");
  }

  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
