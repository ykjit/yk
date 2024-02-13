// # Shadow stack not supported on non-main threads.
// ignore-if: true
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_STATS=-
//   stderr:
//     {
//       ...
//       "trace_executions": 1,
//       "traces_compiled_err": 0,
//       "traces_compiled_ok": 1,
//       "traces_recorded_err": 1,
//       "traces_recorded_ok": 1
//       ...
//     }

// Notice that this test is not itself tested, because of the "ignore" at the
// top!

#include <assert.h>
#include <err.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

struct S {
  YkLocation *loc;
  YkMT *mt;
  uint8_t i;
};

void *main_loop(void *arg) {
  struct S *s = (struct S *)arg;
  while (true) {
    yk_mt_control_point(s->mt, s->loc);
    if (s->i == 0)
      return NULL;
    fprintf(stdout, "i=%d\n", s->i);
    s->i--;
  }
  return NULL;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  // Thread t1 will try tracing the loop and return before a full loop has
  // occurred...
  pthread_t t1;
  struct S t1_data = {&loc, mt, 0};
  if (pthread_create(&t1, NULL, main_loop, (void *)&t1_data) != 0)
    err(EXIT_FAILURE, "pthread_create");
  pthread_join(t1, NULL);

  // ...so when t2 tries tracing the loop it will realise there was a
  // recording error.
  pthread_t t2;
  struct S t2_data = {&loc, mt, 1};
  if (pthread_create(&t2, NULL, main_loop, (void *)&t2_data) != 0)
    err(EXIT_FAILURE, "pthread_create");

  pthread_join(t2, NULL);

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
