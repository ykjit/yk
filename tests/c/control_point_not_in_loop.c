// Compiler:
//   status: error
//   stderr:
//     ...
//     ...error: yk_mt_control_point() must be called inside a loop.
//     ...

// Check that the system crashes if the control point is not in a loop.

#include <stdbool.h>
#include <stdlib.h>
#include <yk.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_control_point(mt, NULL); // Not in a loop!
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
