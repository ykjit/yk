// Compiler:
// Run-time:
//   stderr:
//    ...
//    ret1 = 0xdead
//    ...
//    oof2: {{_}} 20
//    ...
//    ret2 = 0xbaff1ed
//    ...

// This test checks that new threads create their own shadow stack. For this it
// creates two threads `foo` and `bar` in such a way that calling `overwrite`
// on the main thread would assign it the same shadow stack as `bar`, if
// threads do not have their own shadow stacks. Thus creating variables on the
// stack of `overwrite` clobbers values on the stack of `bar`.

#include <pthread.h>
#include <yk.h>
#include <yk_testing.h>
#include <unistd.h>
#include <stdatomic.h>
#include <time.h>

atomic_int order;
struct timespec ts;

void overwrite() {
  int a = 99;
  int b = 99;
  int c = 99;
  int d = 99;
  int e = 99;
  int f = 99;
  int g = 99;
  int h = 99;
  fprintf(stderr, "explode: %p %p %p %p %p %p %p %p\n", &a, &b, &c, &d, &e, &f, &g, &h);
  // Now that we've clobbered `bar`'s stack, hand control back to `bar`.
  atomic_store(&order, 3);
}

void *foo(void *arg) {
  int i = 10;
  fprintf(stderr, "foo1: %p %d\n", &i, i);
  // Hand control back to `main` so it can call `bar`.
  atomic_store(&order, 1);
  while(atomic_load(&order) != 2) {
      nanosleep(&ts, NULL);
  }
  fprintf(stderr, "foo2: %p %d\n", &i, i);
  return (void *) 0xdead;
}

void *bar(void *arg) {
  int i = 20;
  fprintf(stderr, "oof1: %p %d\n", &i, i);
  // Hand control back to `foo`.
  atomic_store(&order, 2);
  while (atomic_load(&order) != 3) {
      nanosleep(&ts, NULL);
  }
  fprintf(stderr, "oof2: %p %d\n", &i, i);
  // Check that the wrapper passes on the return value.
  return (void *) 0xbaff1ed;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 100);
  YkLocation loc = yk_location_new();

  ts.tv_sec = 0;
  ts.tv_nsec = 10000000; // 10ms

  int i = 1;
  atomic_store(&order, 0);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    pthread_t thread;
    pthread_t thread2;
    pthread_create(&thread, NULL, foo, (void *)1);
    while (atomic_load(&order) != 1) {
      nanosleep(&ts, NULL);
    }
    pthread_create(&thread2, NULL, bar, (void *)2);
    void * ret1;
    pthread_join(thread, &ret1);
    fprintf(stderr, "ret1 = %p\n", ret1);
    overwrite();
    void * ret2;
    pthread_join(thread2, &ret2);
    fprintf(stderr, "ret2 = %p\n", ret2);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return 0;
}
