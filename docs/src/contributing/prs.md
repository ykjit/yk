# Pull Requests

We welcome all and any contributions, from code to documentation to examples,
submitted as *pull requests* to one of our [GitHub
repositories](https://github.com/ykjit/). Pull requests are reviewed by a yk
contributor: changes may be requested in an ongoing dialogue between reviewee
and reviewer; and, if and when agreement is reached, the pull request will be
merged by the reviewer.

This page explains yk's approach to the pull request model, based on
observations of good practise in other projects, and our own experiences.
Please note that this is a living process: the details are subject to continual
refinement and change, and no document can perfectly capture all the
possibilities that will crop up in real life.


## What makes a good pull request?

The typical aim of a pull request to a yk repository is to make a focussed
change such as: add a new feature; fix a bug; or edit documentation. Good pull
requests are:

  1. *Self-contained*. A pull request should not try and change everything at
     once. Ideally a pull request fixes a single bug or adds a single feature;
     but sometimes a set of interdependent fixes or features makes more sense.
     In general, small pull requests are better than big pull requests.

  2. *Tested*. Every piece of new functionality, and every bug fix, must have
     at least one accompanying test, unless testing is impractical. Authors
     should also strive not to break any other part of the system. yk runs
     various tests before a pull request is merged: this reduces, but does
     not remove, the chances of a merged pull request breaking expectations:
     careful thought and testing on the part of the pull request author are
     still required.

  3. *Documented*. Code diffs show *what* has been changed, but not *why*.
     Documentation explains to other humans the context for a change (i.e. why
     something needed to be changed), the reason why the change is as it is
     (i.e. could the change have taken a different form? what alternatives were
     considered or attempted?), and the consequences of doing so (i.e. does
     this make something easier or harder in the future?). Documentation comes
     in the form of comments in code, "external" documentation (such as you
     read here), and commit and pull request messages. In whatever form it
     comes, documentation must be clear (as easy to understand as possible),
     concise (as short as possible while getting all the required points
     across), complete (not missing any important points), and necessary (not
     documenting that which is obvious).

  4. *Useful*. The more people that will benefit from a pull request, the more
     likely it is to be merged. The bar for "useful" is set fairly low, but
     that does not mean that every pull request should be merged: for example,
     a pull request which simply changes code to your preferred style is not
     only not useful but may cause problems for other people working in
     parallel. However, porting the code to a little used platform is often
     useful (provided it doesn't cause undue problems for more widely used
     platforms).

  5. *Harmonious*. As far as possible, new code should feel like a natural
     extension of existing code. Following existing naming conventions, for
     example, can sometimes grate, but internal consistency helps those who
     read and edit the code later.


## The reviewer and reviewee covenant

The aim of pull request reviewing is to make yk better while ensuring that our
quality standards are maintained. Ultimately, it is in yk's best interests for
all pull requests which satisfy the criteria listed in the previous section to
be merged in.

Both reviewer and reviewee have their part to play in making this process work.
Most importantly, both parties must assume good faith on the part of the other:
for example, questions are an opportunity to learn or explain, not to attack.
Clear, polite, communication between both parties is required at all times.
Reviewers should respond in a timely manner to comments, while understanding
that reviewees may have many outside responsibilities that mean their responses
are less timely.

Reviewers should help a reviewee meet the expected standards, via questioning
and explicit guidance, without setting the bar unnecessarily high. Reviewers
need to accept:

  * that a pull request cannot solve every problem, and some problems are best
    deferred to a future pull request;
  * and that some variance of individual style is inevitable and acceptable.

Put another way, while we set high standards, and require all contributions to
meet them, we are not foolish enough to expect perfection. In particular,
reviewers must be welcoming to newcomers, who may not be familiar with our
processes or standards, and adjust accordingly.


## The pull request process in yk repositories

To raise a pull request, you must:

  1. Fork the relevant yk GitHub repository to your account. Note that pull
     requests must never come from a branch on a yk repository.

  2. Create a new branch in your fork. If you are unsure about the name, start
     with something generic and rename it later with `git branch -m <new
     name>`.

  3. Make your changes and commit them. It is not only allowed, but often best,
     to have multiple commits: each commit should be a logical change building
     upon one of its predecessors. Each commit should be capable of passing all
     tests successfully (amongst other things, to avoid breaking `git bisect`).

  4. Push your branch to your GitHub fork and raise a pull request. Give the
     pull request a meaningful title (for example, "Fix bug" is not helpful but
     "Deal with an empty list correctly" is) and a description (empty
     descriptions are almost never acceptable). If the pull request fixes all
     aspects of a GitHub issue, add the text `Fixes #<GitHub issue number>`, as
     a line on its own, into the description: GitHub will then automatically
     close that issue when the pull request is merged.

Your pull request has now been raised and will be reviewed by a yk contributor.
The aim of a review is two fold: to ensure that contributions to yk are of an
acceptable quality; and that at least two people (the reviewer and reviewee)
understand what was changed and why.

The reviewer will comment in detail on specific lines of the commit and on the
overall pull request. Comments might be queries (e.g. "can the list ever be
empty at this point?") or corrections (e.g. "we need to deal with the
possibility of an empty list at this point"). For each comment you can either:

  1. Address the issue. For example, if the reviewer has pointed out correctly
     that something needs to be changed then:
       1. make a small additional commit (one per fix);
       2. push it to your branch;
       3. and, in the same place as the reviewer made their comment, add a new
	  comment `Fixed in <git hash>`.

     The reviewer will then review your change and either: mark the
     conversation as "resolved" if their point is adequately addressed; or
     raise further comments otherwise.

     Note that the reviewee must not: mark conversations as resolved (only
     the reviewer should do so); force push changes during this stage (push
     new commits instead).

  2. Add a query. You might not understand the reviewer's question: it's fine
     to ask for clarification.

  3. Explain why you think there is not an issue. The reviewer might have
     misunderstood either the details of a change, or the pull request's
     context (e.g. the pull request might not have intended to fix every
     possible issue). A complete, and polite explanation, can help both
     reviewer and reviewee focus on the essentials.

Often multiple rounds of comments, queries, and fixes are required. When the
reviewer is happy with the changes, they will ask the reviewee to "please
squash" their pull request. See [squashing and
rebasing](#squashing-and-rebasing) for guidance on this.

Once squashing and rebasing has occurred, the reviewer will then push the
button to merge the PR: continuous integration will then run, which may spot
issues, and cause the merge to fail. If so, please use the log generated to fix
the problem. If the fix is trivial -- e.g. a forgotten `cargo fmt` -- it is
acceptable to force push a fix. Note that unrequested force pushes such as this
must not rebase against a different commit at the same time --- doing so makes
it impossible to review the changed PR, and the reviewer will have to ask the
reviewee to undo the force push. If in doubt as to whether a fix is simple
enough to count "trivial", push new commits rather than force pushing.


## Squashing and rebasing

The aim of squashing and rebasing is to provide a readable sequence of
commits for those who later need to look at the history of changes to a
particular part of the repository. The general aim is to provide a pleasing
sequence of individually documented commits which clearly articulate a change
to future readers.

At the very least, all of the "fix commits" must be merged away: commonly,
these are merged into the main commits, or, though rarely, into a new
commit(s). It is not required, and is often undesirable, to squash all commits
down into a single commit: when multiple commits can better explain a change,
they are much preferred. During squashing, you should also check that commit
messages still accurately document their contents: revise those which need
updating.

The process of squashing is synonymous with `git rebase`, so when you are asked
to squash it is also acceptable to: rebase the commit against the `master`
branch at the same time; and force push the resulting rebased branch.


## Documentation

Most of us prefer programming to writing documentation, whether that
documentation is in the form of a comment or commit description (etc). That can
lead us to rush, or omit, documentation, making life difficult for pull request
reviewers and future contributors. We must prioritise creating and maintaining
documentation.

yk's documentation must be clear, concise, and complete. A good rule of thumb
is to write documentation to the quality you would like to read it: use
"proper" English (e.g. avoid colloquialisms), capitalise and punctuate
appropriately, and don't expect to phrase things perfectly on your first
attempt.

In the context of pull requests, bear in mind that the code captures *what* has
been changed, but not *why*: good documentation explains the context of a
change, the reason the change is as it is, and the consequences of the change.
The pull request itself should come with a description of what it aims to
achieve and each individual commit must also contain a self-contained
description of its changes.


## Formatting

yk's continuous integration setup only accepts contributions of well-formatted
code. Changed Rust code must be formatted with `cargo fmt`. Changed C++ code
must be formatted with `cargo xtask cfmt`. Please run both before you raise PR,
and rerun them each time you are about to make a commit in response to reviewer
comments.


## Automated testing

Before pull requests are merged into yk they must pass automated tests. yk uses
[bors](https://bors.tech/) and [Buildbot](https://www.buildbot.net/) to run the
`.buildbot.sh` file in yk's root in a fresh Docker image: if that file executes
successfully, the pull request is suitable for merging. Pull requests may edit
`.buildbot.sh` as with any other file, though one must be careful not to slow
down how long it takes to run unduly. In general, only yk contributors can
issue `bors` commands, though they can in certain situations give external
users the right to issue commands on a given pull request. Users given this
privilege should use it responsibly.
