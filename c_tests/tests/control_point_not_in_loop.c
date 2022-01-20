// Compiler:
//   status: error
//   stderr:
//     ...
//     ...error: yk_control_point() must be called inside a loop.
//     ...

// Check that the system crashes if the control point is not in a loop.

#include <stdlib.h>
#include <stdbool.h>
#include <yk.h>

int main(int argc, char **argv) {
  yk_control_point(NULL); // Not in a loop!
  return (EXIT_SUCCESS);
}
