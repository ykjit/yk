// Run-time:
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-tracing: start-tracing
//     ...
//     yk-tracing: stop-tracing
//     ...
//     yk-execution: enter-jit-code
//     ...

// Testing that when all interpreter state is in globals the JIT produces
// a trace block with no Arg instructions.
// Previously this caused an unconditional panic in p_block.

#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

long g_pc = 0;
YkMT *g_mt = NULL;
YkLocation i;

__attribute__((yk_outline)) static void debug_fn(void) {}

int main(void) {
  g_mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(g_mt, 0);
  yk_mt_sidetrace_threshold_set(g_mt, 4);
  i = yk_location_new();

  for (; g_pc < 20; g_pc++) {
    yk_mt_control_point(g_mt, &i);
    debug_fn();
    if (g_pc >  15) {
      break;
    }
  }

  yk_location_drop(i);
  yk_mt_shutdown(g_mt);
  return EXIT_SUCCESS;
}
