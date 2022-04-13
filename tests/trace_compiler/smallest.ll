; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;        ret {{type}}...
;      }
;      --- End jit-pre-opt ---

define void @main() {
entry:
    unreachable
}
