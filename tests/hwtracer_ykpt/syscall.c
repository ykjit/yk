// Run-time:
//   stderr:
//     hello

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yk_testing.h>

#define THE_STRING "hello\n"

int main(void) {
  size_t remain = strlen(THE_STRING);
  NOOPT_VAL(remain);

  void *tc = __hwykpt_start_collector();

  while (remain) {
    int wrote = write(STDERR_FILENO, THE_STRING, strlen(THE_STRING));
    if (wrote == -1)
      exit(EXIT_FAILURE);
    remain -= wrote;
  }

  void *trace = __hwykpt_stop_collector(tc);
  __hwykpt_libipt_vs_ykpt(trace);
}
