.global call_check_csrs
.intel_syntax noprefix

call_check_csrs:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov ebx, 0x12345
    mov r12d, 0x23456
    mov r13d, 0x34567
    mov r14d, 0x45678
    mov r15d, 0x56789

    call interp

    push rax

    cmp rbx, 0x12345
    jne 1f
    cmp r12, 0x23456
    jne 1f
    cmp r13, 0x34567
    jne 1f
    cmp r14, 0x45678
    jne 1f
    cmp r15, 0x56789
    jne 1f

    pop rax
    jmp 2f

1:
    pop rax
    mov eax, -1

2:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
