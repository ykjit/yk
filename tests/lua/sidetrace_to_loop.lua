-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YK_SIDETRACE_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     h1
--     yk-jit-event: start-tracing: sidetrace_to_loop.lua:35: GTI
--     yk-jit-event: stop-tracing: sidetrace_to_loop.lua:35: GTI
--     --- Begin debugstrs: header: sidetrace_to_loop.lua:35: GTI ---
--       sidetrace_to_loop.lua:36: TEST
--       sidetrace_to_loop.lua:36: JMP
--       sidetrace_to_loop.lua:45: ADDI
--       sidetrace_to_loop.lua:45: JMP
--       sidetrace_to_loop.lua:35: GTI
--     --- End debugstrs ---
--     h2
--     yk-jit-event: enter-jit-code: sidetrace_to_loop.lua:35: GTI
--     yk-jit-event: deoptimise
--     yk-jit-event: enter-jit-code: sidetrace_to_loop.lua:35: GTI
--     yk-jit-event: deoptimise
--     h3
--     yk-jit-event: enter-jit-code: sidetrace_to_loop.lua:35: GTI
--     yk-jit-event: deoptimise
--     yk-jit-event: start-side-tracing: sidetrace_to_loop.lua:35: GTI
--     yk-warning: tracing-aborted: tracing unrolled a loop: sidetrace_to_loop.lua:40: FORLOOP
--     yk-jit-event: enter-jit-code: sidetrace_to_loop.lua:35: GTI
--     yk-jit-event: deoptimise
--     yk-jit-event: start-side-tracing: sidetrace_to_loop.lua:35: GTI
--     exit

function h(i, b1, b2)
  while i > 0 do
    if b1 then
     --
    else
      if b2 then
        for j = 0, 3 do
          --
        end
      end
    end
    i = i - 1
  end
end

io.stderr:write("h1\n")
h(3, true, true)
io.stderr:write("h2\n")
h(1, false, false)
io.stderr:write("h3\n")
h(1, false, true)
io.stderr:write("exit\n")
