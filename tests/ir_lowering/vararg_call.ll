; Dump:
;   stdout:
;     ...
;     func f(%arg0: i32, ...) -> i32 {
;     ...
;     func main(...
;     bb0:
;       %0_0: i32 = arg(0)
;       ...
;       %0_2: i32 = call f(%0_0) [safepoint: 1i64, ()]
;     ...


; Check that varargs calls lower properly.

declare void @llvm.experimental.stackmap(i64, i32, ...);
define i32 @f(i32 %0, ...) noinline { ret i32 6 }

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call i32 (i32, ...) @f(i32 %argc)
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 1, i32 0);
  ret i32 1
}
