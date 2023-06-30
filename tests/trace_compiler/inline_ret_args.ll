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
;        %{{1}} = sub i32 %{{0}}, 2
;        %{{2}} = add i32 %{{1}}, 3
;        br label %loopentry
;      }
;      ...
;      --- End jit-pre-opt ---

; Check that inlining a function with arguments and a return value works.

define i32 @f(i32 %0, i32 %1) {
    %3 = sub i32 %0, %1
    ret i32 %3
}

define void @main() {
entry:
    %0 = add i32 1, 1
    %1 = call i32 @f(i32 %0, i32 2)
    call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0, i32 %0)
    %2 = add i32 %1, 3
    unreachable
}
declare void @llvm.experimental.stackmap(i64, i32, ...)
