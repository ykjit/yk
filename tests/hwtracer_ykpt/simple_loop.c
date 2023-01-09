// Run-time:

#include <stdio.h>
#include <yk_testing.h>

int main(void) {
  int i = 4;
  int x = 0;
  NOOPT_VAL(i);
  NOOPT_VAL(x);

  void *tc = __hwykpt_start_collector();
  while (i > 0) {
    printf("i=%d, x=%d\n", i, x);
    x += 2;
    i--;
  }
  void *trace = __hwykpt_stop_collector(tc);

  __hwykpt_libipt_vs_ykpt(trace);
}
