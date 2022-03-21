; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt,jit-post-opt
;   env-var: YKT_TRACE_BBS=main:0,call_me:0,main:0
;   stderr:
;     ...
;     --- Begin jit-pre-opt ---
;     ...
;     call void @call_me()...
;     ...
;     --- End jit-pre-opt ---
;     ...
;     --- Begin jit-post-opt ---
;     ...
;     call void @call_me()...
;     ...
;     --- End jit-post-opt ---
;     ...

; Check that the `yk_outline` annotation works.

define void @call_me() #0 {
    ret void
}
attributes #0 = { "yk_outline" "noinline"}

define void @main() {
entry:
    call void @call_me()
    unreachable
}

