// Run-time:
//   env-var: YK_JITC=none
//   stderr:
//     exit

// Check that setting different shadowstack sizes works.

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <yk.h>
#include <yk_testing.h>

struct worker {
  YkMT *mt;
  size_t size;
};

static void *run(void *arg) {
  struct worker *worker = arg;
  YkLocation loc = yk_location_null();
  int iterations = 2;
  NOOPT_VAL(iterations);
  while (iterations-- > 0)
    yk_mt_control_point(worker->mt, &loc);
  int local = 42;
  NOOPT_VAL(local);
  void *start, *end;
  yk_thread_shadowstack_bounds(&start, &end);
  assert((uintptr_t)&local >= (uintptr_t)start);
  assert((uintptr_t)&local < (uintptr_t)end);
  assert((uintptr_t)end - (uintptr_t)start <= worker->size);
  // Write to the end of the allocation to check that it's worked.
  ((volatile unsigned char *)start)[worker->size - 1] = 123;
  NOOPT_VAL(local);
  assert(local == 42);
  return NULL;
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  const size_t sizes[] = {2 * 1024 * 1024, 16 * 1024 * 1024};
  for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    yk_set_default_shadowstack_size(sizes[i]);
    struct worker worker = {mt, sizes[i]};
    pthread_t thread;
    assert(pthread_create(&thread, NULL, run, &worker) == 0);
    assert(pthread_join(thread, NULL) == 0);
  }
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return 0;
}
