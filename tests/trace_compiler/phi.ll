; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,main:1,main:3
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define internal void @__yk_compiled_trace_0(...
;      ...
;      guardfail:...
;      ...
;      {{bb}}:...
;        %{{y}} = add i32 %{{x}}, 1
;        %{{z}} = add i32 %{{y}}, 2
;        ret void
;      }
;      ...
;      --- End jit-pre-opt ---

; Check that compiling a PHI node selects the correct value.

define void @main() {
entry:
    %0 = add i32 0, 0
    %1 = icmp eq i32 %0, 0
    br i1 %1, label %true, label %false

true:
    %2 = add i32 %0, 1
    br label %join

false:
    br label %join

join:
    %3 = phi i32 [%2, %true], [%0, %false]
    %4 = add i32 %3, 2
    unreachable
}
