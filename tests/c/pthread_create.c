// Compiler:
// Run-time:
//   status: success
//   stdout: 1,2,3,0
//   stderr:

#include <pthread.h>
#include <stdio.h>
#include <yk.h>
#include <yk_testing.h>

void *thread_function(void *arg) {
  int *a = (int *)arg;
  printf("%d,", *a);
  return NULL;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int MAX_THREADS = 3;
  int thread_args[3] = {1, 2, 3};
  pthread_t thread_handles[MAX_THREADS];

  int a = 0;

  for (int i = 0; i < MAX_THREADS; i++) {
    yk_mt_control_point(mt, &loc);
    pthread_create(&thread_handles[i], NULL, thread_function, &thread_args[i]);
    pthread_join(thread_handles[i], NULL);
  }

  if (a != 0) {
    fprintf(stderr, "Expected 'a' variable to have value of 0\n");
  } else {
    printf("%d", a);
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return 0;
}
