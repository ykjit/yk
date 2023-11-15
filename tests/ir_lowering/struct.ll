; Dump:
;   stdout:
;     ...
;     func main() {
;       bb0:
;         $0_0: {0: i32, 64: i64} = insertvalue const_struct, 100i32
;         ret
;     }

; Check that a structure type lowers correctly.

%s = type { i32, i64 }

define void @main() {
entry:
  %0 = insertvalue %s zeroinitializer, i32 100, 0
  ret void
}
