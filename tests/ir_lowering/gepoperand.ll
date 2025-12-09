; Dump:
;   stdout:
;     ...
;     func main() {
;       bb0:
;         %0_0: ptr = ptr_add @a, 8
;         %0_1: i32 = load %0_0
;         ret
;     }
;     ...

; Check that GEP operands are rewritten to GEP instructions which in turn are
; lowered to a ptr_add and a load.

@a = dso_local constant [5 x i32] [i32 10, i32 11, i32 12, i32 13, i32 14], align 16

define void @main() {
entry:
  %0 = load i32, ptr getelementptr inbounds ([5 x i32], ptr @a, i64 0, i64 2), align 8
  ret void
}
