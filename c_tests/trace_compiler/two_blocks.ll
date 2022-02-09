; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: TRACE_DRIVER_BBS=main:0,main:1
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define internal void @__yk_compiled_trace_0(...
;        %{{0}} = add i32 1, 1
;        %{{1}} = add i32 2, 2
;        ret void
;      }
;      --- End jit-pre-opt ---

define void @main() {
entry:
    %0 = add i32 1, 1
    br label %bb2

bb2:
    %1 = add i32 2, 2
    unreachable
}
