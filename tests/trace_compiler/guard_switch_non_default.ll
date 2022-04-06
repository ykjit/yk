; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,main:4
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define internal {{type}} @__yk_compiled_trace_0(...
;        %{{4}} = add i32 0, 2
;        %{{5}} = icmp eq i32 %{{4}}, 2
;        br i1 %{{5}}, label %{{6}}, label %guardfail
;
;       guardfail:...
;        ...
;
;      {{6}}:...
;        %{{7}} = add i32 %{{4}}, 1
;        ret {{type}}...
;      }
;      ...
;      --- End jit-pre-opt ---

; Check that an appropriate guard is inserted if a trace takes a non-default
; switch case.

define void @main() {
entry:
    %0 = add i32 0, 2
    switch i32 %0, label %bb_default [i32 0, label %bb_zero
                                      i32 1, label %bb_one
                                      i32 2, label %bb_two]
bb_default:
    unreachable

bb_zero:
    unreachable

bb_one:
    unreachable

bb_two:
    %1 = add i32 %0, 1
    unreachable
}
