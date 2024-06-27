; Dump:
;   stdout:
;     ...
;     %{{_}}: float = fadd 3.5float, %{{_}}
;     %{{_}}: double = fadd 4.35double, %{{_}}
;     ...

; Check that lowering floating point constants works.

define void @main(float %x, double %y) {
entry:
  %a = fadd float 3.5, %x
  %b = fadd double 4.35, %y
  ret void
}
