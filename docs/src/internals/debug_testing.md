# Debugging / Testing

## Compile-time features

### `yk_testing`

The `yk_testing` Cargo feature is enabled whenever the `tests` crate is being
compiled, so a regular `cargo build` in the root of the workspace will enable
the feature (to build *without* the feature enabled, do `cargo build -p
ykcapi`).


## Run-time debugging / testing features

### `YKD_SERIALISE_COMPILATION`

When `YKD_SERIALISE_COMPILATION=1`, calls to `yk_control_point(loc)` will block
while `loc` is being compiled.

This variable is only available when building `ykrt` with the `yk_testing`
Cargo feature enabled.
