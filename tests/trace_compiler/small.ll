; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,main:1
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;      ...
;      loopentry:...
;        %{{0}} = add i32 100, 100
;        br label %loopentry
;      }
;      ...
;      --- End jit-pre-opt ---

define void @main() {
entry:
    br label %bb1

bb1:
    %0 = add i32 100, 100
    unreachable
}
