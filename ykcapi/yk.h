#ifndef YK_H
#define YK_H

#include <setjmp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

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
typedef struct YkLocation {
  uintptr_t state;
} YkLocation;

#if defined(__x86_64)
typedef uint32_t YkHotThreshold;
#else
#error Unable to determine type of HotThreshold
#endif

typedef struct YkMT YkMT;

// Create a new `YkMT` instance. If this fails then:
//   * If `err_msg` is `NULL`, this function will abort.
//   * If `err_msg` is not `NULL`:
//       1. A malloc()d string with an error message explaining the failure
//          will be placed in `*err_msg`. It is the callers duty to free this
//          string.
//       2. `yk_mt_new` will return `NULL`.
YkMT *yk_mt_new(char **err_msg);

// Shutdown this MT instance. Will panic if an error is detected when doing so.
// This function can be called more than once, but only the first call will
// have observable behaviour.
void yk_mt_shutdown(YkMT *);

// Notify yk that a given `YkLocation` is about to be executed, allowing yk to
// determine if a trace for this location exists, or one started or stopped, as
// appropriate. The `YkLocation *` be a non-NULL pointer, though a
// `yk_location_null` value causes this function call to be, in essence, a
// no-op: see the documentation for that function for more details.
void yk_mt_control_point(YkMT *, YkLocation *);

// Set the threshold at which `YkLocation`'s are considered hot.
void yk_mt_hot_threshold_set(YkMT *, YkHotThreshold);

// Set the threshold at which guard failures are considered hot.
void yk_mt_sidetrace_threshold_set(YkMT *, YkHotThreshold);

// Returns `true` if yk is running in the "pure" interpreter i.e. (1) without
// tracing the code and (2) not in JIT compiled code. During trace
// optimisation, calls to this function will be fully optimised away.
bool yk_is_interpreting(void);
#define yk_is_interpreting() \
    ((bool)__builtin_expect((yk_is_interpreting)(), 1))

// Create a new `YkLocation`.
//
// Note that a `YkLocation` created by this call must not simply be discarded:
// if no longer wanted, it must be passed to `yk_location_drop` to allow
// appropriate clean-up.
YkLocation yk_location_new(void);

// Force the location to have a HotLocation and assign it a debug string.
//
// Debug strings (where present) are displayed in some log messages.
//
// The intended use of this is for interpreters to add source-level information
// to locations for ease of debugging and human comprehension.
//
// This function is only avaliable if yk was built with the "ykd" feature.
YkLocation yk_location_set_debug_str(YkLocation *, char *);

// Create a new NULL-equivalent `Location`. Such a `YkLocation` denotes a point
// in a program which can never contribute to a trace.
YkLocation yk_location_null(void);

// Determine if the location is a "null" location.
static inline bool yk_location_is_null(YkLocation l) {
  return l.state == 0;
}

// Clean-up a `Location` previously created by `yk_new_location`. The
// `Location` must not be further used after this call or undefined behaviour
// will occur.
void yk_location_drop(YkLocation);

// Promote a value to a constant. This is a generic macro that will
// automatically select the right `yk_promote` function to call based on the
// type of the value passed.
#define yk_promote(X) ((__typeof__(X)) _Generic((X), \
                               int: __yk_promote_c_int, \
                               unsigned char: __yk_promote_c_unsigned_char, \
                               unsigned int: __yk_promote_c_unsigned_int, \
                               unsigned short: __yk_promote_c_unsigned_short, \
                               long long: __yk_promote_c_long_long, \
                               uintptr_t: __yk_promote_usize, \
                               default: __yk_promote_ptr \
                              )(X))
int __yk_promote_c_int(int);
unsigned char __yk_promote_c_unsigned_char(unsigned char);
unsigned int __yk_promote_c_unsigned_int(unsigned int);
unsigned short __yk_promote_c_unsigned_short(unsigned short);
long long __yk_promote_c_long_long(long long);
// Rust defines `usize` to be layout compatible with `uintptr_t`.
uintptr_t __yk_promote_usize(uintptr_t);
const void *__yk_promote_ptr(const void *);

// Perform an idempotent load of the pointer `*P` returning a value of `P`.
//
// WARNING: idempotent is a _very_ strong guarantee and getting it wrong leads
// to undefined behaviour.
#define yk_idempotent_load(P) ({ \
    __auto_type __yk_p = (P); \
    __builtin_bit_cast(__typeof__(*__yk_p), _Generic( \
        (char (*)[sizeof(*__yk_p)])0, \
        char (*)[1]: __yk_idempotent_load_uint8_t, \
        char (*)[2]: __yk_idempotent_load_uint16_t, \
        char (*)[4]: __yk_idempotent_load_uint32_t, \
        char (*)[8]: __yk_idempotent_load_uint64_t \
    )(__yk_p)); \
})
__attribute__((yk_idempotent)) uint8_t __yk_idempotent_load_uint8_t(const void *);
__attribute__((yk_idempotent)) uint16_t __yk_idempotent_load_uint16_t(const void *);
__attribute__((yk_idempotent)) uint32_t __yk_idempotent_load_uint32_t(const void *);
__attribute__((yk_idempotent)) uint64_t __yk_idempotent_load_uint64_t(const void *);

// Promote a code ID. This has the same externally observable effects as a
// normal call to `yk_promote` but also embeds knowledge that this code ID is
// part of any trace being recorded. From yk's perspective a "code ID" is a
// monotonically increasing integer uniquely identifying a chunk of code. Code
// IDs cannot be reused.
#define yk_promote_code_id(X) __yk_promote_code_id(X)
uint64_t __yk_promote_code_id(uint64_t);

// Inform yk that a code ID is not at the moment, and will never in the future,
// be in use: yk uses this to free up memory (etc).
//
// This must be called at most once for a given code ID. Note: it is
// safe to call this for a code ID even if `yk_promote_code_id` has not been
// called with that code ID.
void yk_retire_code_id(YkMT *, uint64_t);

/// Associate a UTF-8 compatible string to the next instruction to be traced.
/// The string will be copied by this function, so callers can safely reuse the
/// memory after this call has completed.
void yk_debug_str(char *);

int yk_pthread_create(
    pthread_t *restrict thread,
    const pthread_attr_t *restrict attr,
    void *(*start_routine)(void *),
    void *restrict arg
);

/// Interpreters must call this instead of `longjmp` to ensure that yk can,
/// when tracing, deal properly with the effects of `longjmp`.
__attribute__((noreturn))
void yk_longjmp(jmp_buf env, int val);

// Set the process-wide shadow stack size (in bytes) for future pthreads.
void yk_set_default_shadowstack_size(size_t size);

// Call `closure` for each running thread's shadowstack. It is guaranteed that
// the shadowstack passed to `closure` on each call is valid for the duration
// of `closure`. `closure` must therefore not start additional threads / kill
// threads (etc).
void yk_foreach_shadowstack(void (*closure)(void* shadow_start, void* shadow_end));

void yk_thread_shadowstack_bounds(void **start, void **end);

#endif
