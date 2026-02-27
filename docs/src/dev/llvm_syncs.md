# LLVM syncs

The compiler component of the system, ykllvm, is a fork of LLVM. Every so often
we merge upstream LLVM changes into ykllvm to inherit new features and
optimisations. It's also a good idea to sync so that modern tools (e.g. cmake)
still work with it.

This page contains some tips on how to do an LLVM sync.

## Minimise the diff to upstream first

Before attempting an upstream sync, minimise (as much as is realistically
possible) the diff between ykllvm and the last time llvm was merged into
ykllvm.

For example, if the previous LLVM sync merged in upstream commit `1234abcd`

```
git diff 1234abcd  -- ':(exclude).github/' ':(exclude).buildbot.sh'
```

Look for unnecessary deviations from upstream and raise a PR to revert those
first. This will minimise merge conflicts.

## Consider using `rerere` in git

You are likely to mess up and `rerere` could save you time.

In `~/.gitconfig`:
```
[rerere]
    enabled = true
```

## Consider chunking your sync up into smaller, more manageable units of work

LLVM moves very fast. You are likely to get lots of merge conflicts,
compilation and logic errors with an LLVM sync. You have git at your disposal,
so you might consider doing several smaller syncs, especially if you are years
behind upstream.

For example, instead of syncing 18 months worth of LLVM commits in one go, you
can do 3 syncs, each with 6 months worth of upstream commits.

## What commit to merge

The way the LLVM release process works, when a release is being prepared, a
branch is created into which fixes are included. Eventually a release tag is
made.

Although tempting, it's better to not sync to release tags, because this means
subsequent syncs will have divergent histories and that can cause problems.

LLVM is pretty well-tested, so it's generally safe to choose an arbitrary
commit from the `main` branch and merge that into ykllvm's main branch.

## Doing the merge and dealing with fallout.

When you are ready, to start the merge:

```sh
git merge <upstream-commit>
```

Then we have to deal with fallout.

### 1. Get ykllvm building

First deal with merge conflicts:

 - We deleted upstream's `.github`, so if they changed that directory, it will
    make merge conflicts. Right of the bat, you can safely do `git rm .github`
    to resolve those.

 - Forget LLVM tests for now. Just get the thing building first.

 - Go through conflict files one-by-one resolving the problems as best you can.
   You may have to make some guesses and rely on the later testing stages to
   tell you if you got it right.

### 2. yk tests

Once ykllvm builds, run the `yk` tests using the new ykllvm. Chances are things
break. Fix those.

Then run `tryci` over the `yk` repo. Fix any problems.

### 3. LLVM tests

Then go back to ykllvm and fix the merge conflicts in the tests and any failing
tests.

 - To run all tests: `cmake --build build --target check-all`
 - To run an individual test `./path/to/llvm-lit /path/to/test/file`
   - add `-vv` to see test output and what didn't match.
 - fixing stackmap tests, particularly MIR tests, can be difficult. If the test
   in question isn't for a platform `yk` supports, it's OK to mark it `XFAIL`
   to skip it.

To `XFAIL` a test, add to the top of the test file:

```
; yk: <a comment saying why you did this>
; XFAIL: *
```

Note the comment character at the start of the above lines may need to change
depending on the kind of input file.

### 4. Benchmarks

`tryci` will have tested short benchmarking runs (for soundness only), but you
should now benchmark the system before and after your sync to check for obvious
performance fallout.

Large performance regressions can occur if LLVM starts emitting bytecode that
the yk AOT IR serialiser can't handle. Run the slow benchmark with `YKD_LOG=2`
and check for traces being aborted due to unimplemented instructions.

Diffing AOT and trace IR may be able to tell you why performance has changed.

Once all is well, you can raise PRs. Good luck!
