// Run-time:
//   stderr:
//     ggg

#include <inttypes.h>
#include <stdio.h>
#include <yk_testing.h>

__attribute__((noinline)) void f(void) { fprintf(stderr, "fff\n"); }

__attribute__((noinline)) void g(void) { fprintf(stderr, "ggg\n"); }

int main(void) {
  void (*callee)(void) = &g;
  NOOPT_VAL(callee);
  void *tc = __hwykpt_start_collector();
  callee(); // indirect call
  void *trace = __hwykpt_stop_collector(tc);
  __hwykpt_libipt_vs_ykpt(trace);
}
