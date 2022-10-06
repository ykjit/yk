; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,f:0,main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;      entry:
;        %{{0}} = add i32 1, 1
;        %{{1}} = add i32 %{{0}}, 2
;        ret {{type}}...
;      }
;      ...
;      --- End jit-pre-opt ---

; Check that inlining a trivial function works.

define void @f() {
    ret void
}

define void @main() {
entry:
    %0 = add i32 1, 1
    call void @f()
    %1 = add i32 %0, 2
    unreachable
}
