#include <stdint.h>

uint64_t
add6(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f)
{
    return a + b + c + d + e + f;
}

uint64_t
add_some(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
    (void) c;
    (void) e;
    return a + b + d;
}
