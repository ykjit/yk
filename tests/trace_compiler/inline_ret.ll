; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,f:0,main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;        %{{0}} = add i32 1, 1
;        %{{2}} = add i32 %{{0}}, 5
;        ret {{type}}...
;      }
;      --- End jit-pre-opt ---

; Check that inlining a simple function with a return value works.

define i32 @f() {
    ret i32 5
}

define void @main() {
entry:
    %0 = add i32 1, 1
    %1 = call i32 @f()
    %2 = add i32 %0, %1
    unreachable
}
