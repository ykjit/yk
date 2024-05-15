; Dump:
;   stdout:
;     ...
;     func main(...
;       bb0:
;         unimplemented <<  %{{4}} = getelementptr i32, <8 x ptr> %{{1}}, i32 1>>
;         br bb1
;       bb1:
;         unimplemented <<  %{{6}} = alloca inalloca i32, align 4>>
;         unimplemented <<  %{{7}} = alloca i32, align 4, addrspace(4)>>
;         unimplemented <<  %{{8}} = alloca i32, i32 %2, align 4>>
;         unimplemented <<  %{{9}} = alloca i32, i66 -36893488147419103232, align 4>>
;         br bb2
;      bb2:
;         unimplemented <<  %{{13}} = fadd nnan float %{{3}}, %{{3}}>>
;         unimplemented <<  %{{14}} = udiv exact i32 %{{2}}, 1>>
;         unimplemented <<  %{{15}} = add <4 x i32> %{{44}}, %{{44}}>>
;         ret
;     }

; This test ensures that as-yet unsupported variants of LLVM instructions are
; serialised as an unsupported instruction in the AOT IR. This prevents the JIT
; from silently miscompiling things we haven't yet thought about.

@arr = global [4 x i8] zeroinitializer

define void @main(ptr %ptr, <8 x ptr> %ptrs, i32 %num, float %flt, <4 x i32> %vecnums) {
geps:
  ; note `getelementptr inrange` cannot appear as a dedicated instruction, only
  ; as an inline expression. Hence no check for that in instruction form.
  %0 = getelementptr i32, <8 x ptr> %ptrs, i32 1
  br label %allocas
allocas:
  %1 = alloca inalloca i32
  %2 = alloca i32, addrspace(4)
  %3 = alloca i32, i32 %num
  %4 = alloca i32, i66 36893488147419103232 ; 2^{65}
  br label %binops
binops:
  %5 = fadd nnan float %flt, %flt
  %6 = udiv exact i32 %num, 1
  %7 = add <4 x i32> %vecnums, %vecnums
  ret void
}
