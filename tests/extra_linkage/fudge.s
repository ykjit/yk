.global fudge_return_address
.intel_syntax

# Function that returns to the address stored in its first argument.
fudge_return_address:
    pop rax
    push rdi
    ret
