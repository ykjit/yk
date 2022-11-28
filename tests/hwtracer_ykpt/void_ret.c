// Run-time:

#include <stdio.h>
#include <yk_testing.h>

void __attribute__((noinline)) f() {
  fputs("inside f\n", stderr);
  return;
}

int main(void) {
  int i = 4;
  NOOPT_VAL(i);

  void *tc = __hwykpt_start_collector();
  while (i-- > 0)
    f();
  void *trace = __hwykpt_stop_collector(tc);

  __hwykpt_libipt_vs_ykpt(trace);
}
