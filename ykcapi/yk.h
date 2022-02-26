#ifndef YK_H
#define YK_H

#include <stdint.h>

// A `Location` stores state that the meta-tracer needs to identify hot loops
// and run associated machine code. This is a C mirror of `ykrt::Location`.
//
// Each position in the end user's program that may be a control point (i.e.
// the possible start of a trace) must have an associated `Location`. The
// `Location` does not need to be at a stable address in memory and can be
// freely moved.
//
// Program positions that can't be control points don't need an associated
// `Location`. For interpreters that can't (or don't want) to be as selective,
// a simple (if moderately wasteful) mechanism is for every bytecode or AST
// node to have its own `Location` (even for bytecodes or nodes that can't be
// control points).
typedef struct {
  uintptr_t state;
} YkLocation;

#if defined(__x86_64)
typedef uint32_t YkHotThreshold;
#else
#error Unable to determine type of HotThreshold
#endif

typedef struct YkMT YkMT;

// Create a new `YkMT` instance.
YkMT *yk_mt_new();

// Drop a `YkMT` instance. This must be called at most once per `YkMT`
// instance: calling this function more than once on a `YkMT` instance leads to
// undefined behaviour.
void yk_mt_drop(YkMT *);

// Notify yk that an iteration of an interpreter loop is about to start. The
// argument passed uniquely identifies the current location in the user's
// program. A call to this function may cause yk to start/stop tracing, or to
// execute JITted code.
void yk_control_point(YkMT *, YkLocation *);

// Set the threshold at which `YkLocation`'s are considered hot.
void yk_hot_threshold_set(YkMT *, YkHotThreshold);

// Create a new `Location`.
//
// Note that a `Location` created by this call must not simply be discarded:
// if no longer wanted, it must be passed to `yk_location_drop` to allow
// appropriate clean-up.
YkLocation yk_location_new(void);

// Clean-up a `Location` previously created by `yk_new_location`. The
// `Location` must not be further used after this call or undefined behaviour
// will occur.
void yk_location_drop(YkLocation);

void yk_stopgap(void *addr, uintptr_t size, uintptr_t retaddr, void *rsp);

#endif
