#ifndef YK_H
#define YK_H

#include <stdint.h>

/// A meta-tracer.
struct MT;
typedef struct MT MT;

// A `Location` stores state that the meta-tracer needs to identify hot loops
// and run associated machine code.
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

// Return a reference to the global `MT` instance: at any point, there is at
// most one of these per process and an instance will be created if it does not
// already exist.
MT *yk_mt(void);

// Return this `MT` instance's current hot threshold. Notice that this value
// can be changed by other threads and is thus potentially stale as soon as it
// is read.
YkHotThreshold yk_mt_hot_threshold(MT *);

// Attempt to execute a compiled trace for location `loc`. `NULL` may be passed
// to `loc` to indicate that this particular point in the user's program cannot
// ever be the beginning of a trace.
void yk_control_point(MT *, YkLocation *);

// Mark this point in the program as being a safe point for the stopgap
// interpreter to transition back to normal program execution. Typical stopgap
// safe points are at the backwards jump(s) of an interpreter's main for/while
// loop. If, for example, your interpreter uses interpreted gotos, you may want
// to mark multiple locations as stopgap safe.
void yk_stopgap_safe();

// Create a new `Location`.
//
// Note that a `Location` created by this call must not simply be discarded: if
// no longer wanted, it must be passed to `yk_drop_location` to allow
// appropriate clean-up.
YkLocation yk_location_new(void);

// Clean-up a `Location` previously created by `yk_new_location`. The
// `Location` must not be further used after this call or undefined behaviour
// will occur.
void yk_location_drop(YkLocation);

#endif
