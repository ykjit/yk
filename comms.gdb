set logging enabled on
set breakpoint pending on

break ykrt::trace::swt::cp::return_into_unopt_cp
break ykrt::trace::swt::cp::return_into_opt_cp

break __yk_clone_main
# break *0x0000000000202645
# break simple2.c:28