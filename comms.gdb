set logging enabled on
set breakpoint pending on

# break ykrt::trace::swt::cp::debug_return_into_unopt_cp
# break ykrt::trace::swt::cp::debug_return_into_opt_cp

# break __yk_clone_main
break main
# break /home/pd/yk-fork/tests/c/simple.c:44.c:17
# break buffered_vfprintf

# break before cp
break *0x0000000000202b9a
# break after cp
break *0x0000000000202b9d

break ykrt/src/mt.rs:428
