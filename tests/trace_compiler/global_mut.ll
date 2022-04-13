; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;        store i32 1, i32* @g, align 4
;        ret {{type}}...
;      }
;      --- End jit-pre-opt ---

; Check the trace compiler correctly handles a mutated global.

@g = global i32 5

define void @main() {
entry:
    store i32 1, i32* @g
    unreachable
}
