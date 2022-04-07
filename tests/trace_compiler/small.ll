; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define internal {{type}} @__yk_compiled_trace_0(...
;        %{{0}} = add i32 100, 100
;        ret {{type}}...
;      }
;      --- End jit-pre-opt ---

define void @main() {
entry:
    %0 = add i32 100, 100
    unreachable
}
