#ifndef __HWT_PRIVATE_H
#define __HWT_PRIVATE_H

#include <stddef.h>
#include <inttypes.h>
#include <stdbool.h>

enum hwt_cerror_kind {
    hwt_cerror_unused,
    hwt_cerror_unknown,
    hwt_cerror_errno,
    hwt_cerror_ipt,
};

struct hwt_cerror {
    enum hwt_cerror_kind kind; // What sort of error is this?
    int code;                  // The error code itself.
};

void hwt_set_cerr(struct hwt_cerror *, int, int);

#endif
