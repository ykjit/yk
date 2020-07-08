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

1. **Raise PRs**

The change author raises two PRs: one for `yk` and one for `ykrustc`.

2. **yk review**

The reviewer conducts a review of the changes to the `yk` repo first. We get
the branch squashed and ready, but we don't merge it just yet. If we were to
try it would fail. Remember: we need the compiler change in first.

3. **Review ykrustc**

Next the reviewer switches to the `ykrustc` pull request and conducts a review
there. We get this branch squashed and ready to merge, but stop short of
invoking bors.

4. **Prime the CI**

Next the reviewer should ask the PR author to "prime the CI". A commit should
be pushed which overrides the yk dependency to the branch reviewed in stage 2
of this document. This commit should override the compiler's dependencies on
`ykpack`.

The PR author then logs in to the CI server's web interface and manually starts a
build of the ykrustc branch, including the commit pushed in the last step.

5. **Merge the ykpack change**

Assuming the previous step worked, the CI server has now cached a version of
`ykrustc` which uses the new `ykpack`. Now the reviewer can continue to invoke
bors on the branch we reviewed in step 2 and CI will use the newly cached
compiler to build it, before merging the branch.

6. **Merge ykrustc**

The reviewer can now ask the PR author to remove the priming commit introduced
at stage 4 (using a force push) and once this is done, the PR can be merged
with bors.

Then we are done.
