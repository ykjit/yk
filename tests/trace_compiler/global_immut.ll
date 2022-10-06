; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;      entry:
;        %{{0}} = load i32, i32* @g, align 4
;        %{{1}} = add i32 %{{0}}, 1
;        ret {{type}}...
;      }
;      ...
;      --- End jit-pre-opt ---

; Check the trace compiler correctly handles a not-mutated global.

@g = global i32 5

define void @main() {
entry:
    %0 = load i32, i32* @g
    %1 = add i32 %0, 1
    unreachable
}
