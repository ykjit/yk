#include <linux/perf_event.h>

int check(void) {
  // The perf configuration struct version that first supported Intel PT.
  return PERF_ATTR_SIZE_VER5;
}
