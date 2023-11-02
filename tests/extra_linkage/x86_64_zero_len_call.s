.global zero_len_call
.intel_syntax

# A little function that returns the value of RIP via (what Intel calls) a
# "zero-length call".
zero_len_call:
    mov rax, 0
    call y # zero-length return.
y:
    # FIXME: There have to be an equal number of calls and returns or the
    # outliner's frame counter will get confused. We've already done a `call`
    # with no matching `ret`, so now we have to do a `ret` with no matching
    # `call`.
    #
    # Once https://github.com/ykjit/yk/issues/818 is fixed, we can hopefully
    # remove this hack.
    #
    # Note that `push z` doesn't do what you think it would!
    lea rdi, [z]
    push rdi
    ret
z:
    pop rax
    ret
