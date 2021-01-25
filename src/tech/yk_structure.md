# The Structure of the JIT Runtime

The `yk` repo's structure (and build system) is somewhat unusual as the code is
spread over two workspaces. This is because we need to separately compile parts
of the codebase with different configurations.

Code that we expect to be able to hardware trace needs to be compiled with `-C
tracer=hw`. Crucially, since hardware tracing relies on blocks not being
reordered by LLVM, this flag disables optimisations. However, the JIT runtime
will never be traced and therefore we can (and should) optimise this code.
Further we want to be able to optimise the dependencies of the JIT runtime.

To this end, we have two Rust workspaces in the `yk` repo: the "internal"
(optimised) workspace, and the "external" (unoptimised) workspace. The external
workspace then talks to the internal workspace via `extern` functions defined
in the `ykshim` crate.

```
       External Workspace             |       Internal Workspace
--------------------------------------------------------------------

interpreter --\
              |
              +---> ykshim_client --------> ykshim --> JIT runtime
              |
tests  -------/
```

To build the two workspaces, we use `cargo xtask`.

There are a few implementation details to note:

 - Code traced as part of testing needs to reside in the external workspace.
   Parts of `ykshim`'s API surface exist only for testing.

 - Although we build both workspaces with the same compiler, to avoid potential
   ABI-related issues (where adding a `-C` flag to the `rustc` invocation could
   result in ABI skew), the workspaces communicate via the C ABI.

- Similarly, unless explicitly safe (e.g. `std::ffi` types, or `#[repr(C)]`
  types), we shouldn't assume that types with the same definition have the same
  layout in both workspaces. It is however, always safe for one workspace to
  give the other an opaque pointer to an instance of some type as long as the
  receiving workspace never tries to interpret the value as anything but an
  opaque pointer.

- Due to the separate compilation of the workspaces, some code will be
  duplicated. To avoid collisions of unmangled symbols, the internal workspace
  is compiled into a shared object.

- Since `yk` requires use of the abort strategy, no attempt is made to prevent
  unwinding across the C ABI (which would invoke undefined behaviour).
