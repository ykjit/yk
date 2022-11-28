// Run-time:

#include <inttypes.h>
#include <stdio.h>
#include <yk_testing.h>

__attribute__((noinline)) uint64_t work(uint64_t iters) {
  uint64_t sum = 0;
  NOOPT_VAL(iters);
  while (iters)
    sum += iters--;
  printf("%lu\n", sum);
  return sum;
}

int main(void) {
  void *tc = __hwykpt_start_collector();
  work(10000);
  void *trace = __hwykpt_stop_collector(tc);
  __hwykpt_libipt_vs_ykpt(trace);
}
