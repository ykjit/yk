// Compiler:
// Run-time:
//   status: error
//   stderr: ...
//    not yet implemented: Allocate and set thread-local ShadowStack instance. No support for threads yet!
//    ...

#include <pthread.h>
#include <yk.h>
#include <yk_testing.h>

void *thread_function(void *arg) { return NULL; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3;
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    pthread_t thread;
    pthread_create(&thread, NULL, thread_function, NULL);
    pthread_join(thread, NULL);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return 0;
}
