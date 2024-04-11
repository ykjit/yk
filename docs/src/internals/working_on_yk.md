# Working on yk

yk has several features designed to make it easier to work on yk itself. Most
of these are transparent to the developer (e.g. rebuilding ykllvm when needed):
in this page we document those that are not.


## clangd

The `yk` build system generates compilation command databases for use with
clangd. If you want diagnostics and/or completion in your editor (via an LSP),
you will have to configure the LSP to use `clangd` (the automated build system
puts a `clangd` binary into `target/<debug|release>/ykllvm/bin` that you could
use).
