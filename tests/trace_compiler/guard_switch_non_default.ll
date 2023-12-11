; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,main:1,main:5
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;      ...
;      loopentry:...
;        %{{4}} = add i32 0, 2
;        %{{5}} = icmp eq i32 %{{4}}, 2
;        br i1 %{{5}}, label %{{6}}, label %guardfail
;
;       guardfail:...
;        ...
;
;      {{6}}:...
;        %{{7}} = add i32 %{{4}}, 1
;        br label %loopentry
;      }
;      ...
;      --- End jit-pre-opt ---

; Check that an appropriate guard is inserted if a trace takes a non-default
; switch case.

define void @main() {
entry:
    br label %bb1

bb1:
    %0 = add i32 0, 2
    call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, i32 %0)
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
declare void @llvm.experimental.stackmap(i64, i32, ...)
