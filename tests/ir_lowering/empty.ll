; Dump:
;   stdout:
;     # IR format version: 0
;     # Num funcs: 1
;     # Num consts: 1
;     # Num types: 2
;
;     func main {
;       bb0:
;         ret 0i32
;     }

; The simplest test you could write. Checks an empty module lowers correctly.

define i32 @main() {
entry:
  ret i32 0
}
