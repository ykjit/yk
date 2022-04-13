// Compiler:
//   status: error
//   stderr:
//     ...
//     ...error: ykllvm couldn't find the call to `yk_mt_control_point()`
//     ...

// Check that the system crashes if the control point is missing.

#include <stdlib.h>

int main(int argc, char **argv) {
  // Control point is missing!
  return (EXIT_SUCCESS);
}
