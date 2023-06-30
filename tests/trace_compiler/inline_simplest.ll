; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,f:0,main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;      ...
;      define {{type}} @__yk_compiled_trace_0(...
;      ...
;      loopentry:...
;        %{{0}} = add i32 1, 1
;        %{{1}} = add i32 %{{0}}, 2
;        br label %loopentry
;      }
;      ...
;      --- End jit-pre-opt ---

; Check that inlining a trivial function works.

define void @f() {
    ret void
}

define void @main() {
entry:
    %0 = add i32 1, 1
    call void @f()
    call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, i32 %0)
    %1 = add i32 %0, 2
    unreachable
}
declare void @llvm.experimental.stackmap(i64, i32, ...)
