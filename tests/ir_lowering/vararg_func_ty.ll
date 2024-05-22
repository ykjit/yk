; Dump:
;   stdout:
;     ...
;     func f(%arg0: i32, ...) -> i32 {
;     ...

; Check a vararg function type lowers correctly.

declare void @llvm.experimental.stackmap(i64, i32, ...);

define i32 @f(i32 %0, ...) {
    ret i32 %0
}

define i32 @main() {
    %1 = call i32 (i32, ...) @f(i32 1, i32 2, i32 3);
    call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0);
    ret i32 %1
}
