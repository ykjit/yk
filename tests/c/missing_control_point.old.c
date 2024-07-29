// Compiler:
//   status: success
//   stderr:
//     ...
//     ...warning: ykllvm couldn't find the call to `yk_mt_control_point()`
//     ...

// Check that the ykllvm warns if the control point is missing.

#include <stdlib.h>

int main(int argc, char **argv) {
  // Control point is missing!
  return (EXIT_SUCCESS);
}
