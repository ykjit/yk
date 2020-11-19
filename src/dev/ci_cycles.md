# Continuous Integration (CI) Cycles

When working on Yorick, it's possible to create "CI cycles", which require
special handling. This chapter outlines the problem scenario and how we work
around it.

## Dependency Architecture of ykrustc.

There are two main repos for Yorick:

 * `ykrustc`: the compiler.
 * `yk`: the other stuff, including `ykpack`.

`ykpack` is the library that deals with SIR serialisation and de-serialisation.
The compiler uses it to encode SIR into an ELF section, and the JIT runtime
uses it to decode the section.

This leads to a problem: if you change the format of the SIR (change the
serialised types in any way that would change the binary representation once
serialised), then CI cannot succeed. This is because the `yk` repo needs to be
built with a `ykrustc` which uses the new `ykpack`, but `yk` itself contains
the new `ykpack`.

## How do we Break the Cycle?

The change author raises two PRs: one for `yk` and one for `ykrustc`.

The `ykrustc` PR description must have a line in the format:
```
ci-yk: <github-username> <branch>
```

There must be no other text on the line containing this. For example, the line
`ci-yk: jim myfeature` tells the CI to use the `myfeature` branch of
`https://github.com/jim/yk` for the yk dependency.

Then the reviewer can merge `ykrustc` and then `yk`. Note that bors can only
merge one PR at a time, so merging the `ykrustc` PR will not merge the `yk` PR
automatically.

Note that the last successful CI build of ykrustc is used to build `yk`.
