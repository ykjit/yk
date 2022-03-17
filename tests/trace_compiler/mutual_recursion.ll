; Run-time:
;   env-var: YKD_PRINT_IR=jit-pre-opt
;   env-var: YKT_TRACE_BBS=main:0,f:0,f:1,g:0,f:0,f:2,g:0,f:1,f:2,main:0
;   stderr:
;      --- Begin jit-pre-opt ---
;
;      %0 = type { i8 }
;
;      define internal void @__yk_compiled_trace_0(%0* %0, i64* %1, i64 %2) {
;        %4 = icmp eq i32 1, 0
;        br i1 %4, label %guardfail, label %5
;
;      guardfail:                                        ; preds = %3
;        call void (...) @llvm.experimental.deoptimize.isVoid(i64* %1, i64 %2) [ "deopt"() ]
;        ret void
;
;      5:                                                ; preds = %3
;        call void @f(i32 0)
;        ret void
;      }
;
;      declare void @llvm.experimental.deoptimize.isVoid(...)
;
;      declare void @f(i32)
;      --- End jit-pre-opt ---

define void @f(i32 %0) {
    %2 = icmp eq i32 %0, 0
    br i1 %2, label %done, label %recurse
recurse:
    call void @g()
    br label %done
done:
    ret void
}

define void @g() {
    call void @f(i32 0)
    ret void
}

define void @main() {
entry:
    call void @f(i32 1)
    ret void
}
