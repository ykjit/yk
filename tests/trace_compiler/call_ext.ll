; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;      ...
;      loopentry:...
;        %{{uid}} = call i32 @getuid()
;        br label %loopentry
;      }
;
;      declare i32 @getuid()
;      ...
;      --- End jit-pre-opt ---

; Check that calls to external functions are properly declared in the trace.

declare dso_local i32 @getuid()

define void @main() {
entry:
    %0 = call i32 @getuid()
    unreachable
}
