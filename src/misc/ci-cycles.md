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

We break the cycle using a 3-PR model as follows:

1. **Raise the first two PRs**

The change author raises two PRs: one for `yk` and one for `ykrustc`.

2. **yk review**

The reviewer conducts a review of the changes to the `yk` repo first. We get
the branch squashed and ready, but we don't merge it just yet. If we were to
try it would fail. Remember: we need the compiler change in first.

3. **Review ykrustc**

Next the reviewer switches to the `ykrustc` pull request and conducts a review
there. We get this branch squashed and ready to merge, but stop short of
invoking bors.

4. **Cycle breaker commit**

Next the reviewer should ask the PR author to push a "cycle breaker" commit to
`ykrustc`. This commit overrides the compiler's dependency on `ykpack`, making
it instead use the commit hash for the head of the branch we reviewed in stage
2.

[Here's an example cycle breaker commit](https://github.com/softdevteam/ykrustc/commit/abd1c2b7669c4ab6be8f9a9e6c1704a7e70c2088)

5. **Merge ykrustc**

Once the cycle breaker commit is added, the reviewer can invoke bors to have
the compiler change merged.

6. **Merge the ykpack change**

Assuming the previous step worked, the CI server has now cached a version of
`ykrustc` which uses the new `ykpack`. Now we can continue to invoke bors on
the branch we reviewed in step 2 and CI will use the newly cached compiler to
build it, before merging the branch.

7. **Undo the cycle breaker commit**

Now that the `ykpack` changes are on the master branch of `yk`, we can revert
the cycle breaker commit we introduced to `ykrustc` in step 4. The change
author raises a new PR for this.
