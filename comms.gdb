set logging enabled on
set breakpoint pending on

break ykrt::trace::swt::cp::debug_return_into_unopt_cp
break ykrt::trace::swt::cp::debug_return_into_opt_cp

break __yk_clone_main
break main
break /home/pd/yk-fork/tests/c/simple.c:44.c:17