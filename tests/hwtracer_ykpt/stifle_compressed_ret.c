// Run-time:
//   stderr:
//     before
//     always reached

#include <assert.h>
#include <stdio.h>
#include <yk_testing.h>

// This function returns to the address passed in, which in turn prevents Intel
// PT from compressing the return back to main.
extern void fudge_return_address(void *);

int main(void) {
  void *ra = &&ret_here;
  NOOPT_VAL(ra);

  fprintf(stderr, "before\n");
  void *tc = __hwykpt_start_collector();
  fudge_return_address(ra);
  fprintf(stderr, "never reached\n");

ret_here:
  fprintf(stderr, "always reached\n");
  void *trace = __hwykpt_stop_collector(tc);

  __hwykpt_libipt_vs_ykpt(trace);
}
