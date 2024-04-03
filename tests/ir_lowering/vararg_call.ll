; Dump:
;   stdout:
;     ...
;     func f($arg0: i32, ...) -> i32 {
;     ...
;     func main(...
;     bb0:
;       $0_0: i32 = call f($arg0)
;     ...


; Check that varargs calls lower properly.

define i32 @f(i32 %0, ...) noinline { ret i32 6 }

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call i32 (i32, ...) @f(i32 %argc)
  ret i32 1
}
