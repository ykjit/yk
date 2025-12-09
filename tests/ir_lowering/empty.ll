; Dump:
;   stdout:
;     # IR format version: 0
;     # Num funcs: 2
;     # Num consts: 0
;     # Num global decls: 0
;     # Num types: 4
;
;     func main() {
;       bb0:
;         ret
;     }
;     ...

; The simplest test you could write. Checks an empty module lowers correctly.

define void @main() {
entry:
  ret void
}
