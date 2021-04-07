# The Structure of the JIT Runtime

The `yk` repo's structure (and build system) is somewhat unusual as the code is
spread over different workspaces. This is because we need to separately compile
parts of the codebase with different configurations.

Code that we expect to be able to hardware trace needs to be compiled with `-C
tracer=hw`. Crucially, since hardware tracing relies on blocks not being
reordered by LLVM, this flag disables optimisations. However, the JIT runtime
will never be traced and therefore we can (and should) optimise this code.
Further we want to be able to optimise the dependencies of the JIT runtime.

To this end, we have three Rust workspaces in the `yk` repo:
 - The top-level xtask workspace: our build system.
 - The `untraced` workspace: the "always optimised" JIT runtime.
 - The `traced` workspace: the Yorick API intended for use by interpreter authors
   (and associated tests). This is built without optimisations if hardware
   tracing is used.

The `traced` workspace then talks to the `untraced` workspace via an API:

```
            Traced Workspace                  Untraced Workspace
+------------------------------------+ +--------------------------------+
| Interpreter +                      | |                                |
|             |                      | |                                |
|             +----> untraced_api +------> to_traced +----> JIT runtime |
|             |                      | |                                |
|       Tests +                      | |                                |
+------------------------------------+ +--------------------------------+
```

There are a few implementation details to note:

- Code traced as part of testing needs to reside in the `traced` workspace.
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
  duplicated. To avoid collisions of unmangled symbols, the `untraced`
  workspace is compiled into a shared object.

- Since `yk` requires use of the abort strategy, no attempt is made to prevent
  unwinding across the C ABI (which would invoke undefined behaviour).

- The workspace model means that rust-analyzer won't work properly by default.
  See [here](../dev/getting_started.md#rust-analyzer) for details.
