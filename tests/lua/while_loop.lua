-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     0
--     1
--     2
--     yk-jit-event: start-tracing: while_loop.lua:35: LEI
--     3
--     yk-jit-event: stop-tracing: ...
--     --- Begin debugstrs: header: while_loop.lua:35: LEI ---
--       while_loop.lua:36: GETTABUP
--       while_loop.lua:36: GETFIELD
--       while_loop.lua:36: SELF
--       while_loop.lua:36: GETTABUP
--       while_loop.lua:36: MOVE
--       while_loop.lua:36: CALL
--       while_loop.lua:36: LOADK
--       while_loop.lua:36: CALL
--       while_loop.lua:37: ADDI
--       while_loop.lua:37: JMP
--       while_loop.lua:35: LEI
--     --- End debugstrs ---
--     4
--     yk-jit-event: enter-jit-code: while_loop.lua:35: LEI
--     5
--     6
--     yk-jit-event: deoptimise
--     exit

local x = 0
while x <= 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")
