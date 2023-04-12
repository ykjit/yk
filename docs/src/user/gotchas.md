# Gotchas

Here are some potential "gotchas" for interpreter authors:

 - Yk can only currently work with "simple interpreter loop"-style interpreters
   and cannot yet handle unstructured interpreter loops (e.g. threaded
   dispatch).

 - Yk currently doesn't handle calls to `pthread_exit()` gracefully ([more
   details](https://github.com/ykjit/yk/issues/525)).

 - You cannot valgrind an interpreter that is using Intel PT for tracing ([more
   details](https://github.com/ykjit/yk/issues/177)).
