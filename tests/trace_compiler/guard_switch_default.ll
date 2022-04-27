; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,main:1
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;        %{{4}} = add i32 0, 999
;        switch i32 %{{4}}, label %{{5}} [
;          i32 0, label %guardfail
;          i32 1, label %guardfail
;          i32 2, label %guardfail
;        ]
;
;      guardfail:                                        ; preds = %4, %4, %4
;        ...
;
;      {{5}}:...
;        %{{6}} = add i32 %{{4}}, 2
;        ret {{type}}...
;        ...
;      --- End jit-pre-opt ---

; Check that an appropriate guard is inserted if a trace takes the default
; switch case.

define void @main() {
entry:
    %0 = add i32 0, 999
    switch i32 %0, label %bb_default [i32 0, label %bb_zero
                                      i32 1, label %bb_one
                                      i32 2, label %bb_two]
bb_default:
    %1 = add i32 %0, 2
    unreachable

bb_zero:
    unreachable

bb_one:
    unreachable

bb_two:
    unreachable
}
