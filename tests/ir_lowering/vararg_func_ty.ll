; Dump:
;   stdout:
;     ...
;     func f($arg0: i32, ...) -> i32 {
;     ...

; Check a vararg function type lowers correctly.

define i32 @f(i32 %0, ...) {
    ret i32 %0
}

define i32 @main() {
    %1 = call i32 @f(i32 1, i32 2, i32 3);
    ret i32 %1
}
