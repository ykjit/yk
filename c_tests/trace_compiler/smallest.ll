; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: TRACE_DRIVER_BBS=main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define internal void @__yk_compiled_trace_0(...
;        ret void
;      }
;      --- End jit-pre-opt ---

define void @main() {
entry:
    unreachable
}
