# Debugging / Testing

## Run-time Debugging / Testing features

### `YKD_NEW_CODEGEN`

When set to `1` forces the JIT to use the new codegen.

This is temporary, and will be removed once the new codegen is production
quality.

### `YKD_SERIALISE_COMPILATION`

When `YKD_SERIALISE_COMPILATION=1`, calls to `yk_control_point(loc)` will block
while `loc` is being compiled.

This variable is only available when building `ykrt` with the `yk_testing`
Cargo feature enabled.
