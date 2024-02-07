; Dump:
;   stdout:
;     # IR format version: 0
;     # Num funcs: 1
;     # Num consts: 0
;     # Num globals: 0
;     # Num types: 2
;
;     func main() {
;       bb0:
;         ret
;     }

; The simplest test you could write. Checks an empty module lowers correctly.

define void @main() {
entry:
  ret void
}
