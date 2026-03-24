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

#include <assert.h>
#include <err.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define NUM_THREADS 8

// `yk_indirect_inline` is to stop this being marked as untraceable, thus
// ensuring it gets a shadow stack.
__attribute__((yk_indirect_inline))
void *f(void *arg) {
  int x = 0;
  NOOPT_VAL(x);
  void *ss_start = NULL, *ss_end = NULL;
  yk_thread_shadowstack_bounds(&ss_start, &ss_end);
  int inbounds = ((void *) &x >= ss_start) && ((void *) &x <= ss_end);
  fprintf(stderr, "&x=%p, ss=[%p, %p], onshadow=%d\n", &x, ss_start, ss_end, inbounds);
  assert(ss_start < ss_end);
  assert(inbounds);
  return NULL;
}

int main() {
  pthread_t threads[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    if (pthread_create(&threads[i], NULL, f, NULL) != 0)
      err(EXIT_FAILURE, "pthread_create");
  }

  void *thread_res = 0;
  for (int i = 0; i < NUM_THREADS; i++) {
    if (pthread_join(threads[i], &thread_res) != 0)
      err(EXIT_FAILURE, "pthread_join");
  }

  return (EXIT_SUCCESS);
}
