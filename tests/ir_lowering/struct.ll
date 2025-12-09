; Dump:
;   stdout:
;     ...
;     func main(%arg0: {0: i32, 64: i64}) {
;       bb0:
;         %0_0: {0: i32, 64: i64} = arg(0)
;         %0_1: {0: i32, 64: i64} = insert_val %0_0, 100i32
;         ret
;     }
;     ...

; Check that a structure type lowers correctly.

%s = type { i32, i64 }

define void @main(%s %val) {
entry:
  %0 = insertvalue %s %val, i32 100, 0
  ret void
}
