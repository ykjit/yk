-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     0
--     1
--     2
--     yk-tracing: start-tracing: while_loop.lua:34: LEI
--     3
--     yk-tracing: stop-tracing: ...
--     --- Begin debugstrs: header: while_loop.lua:34: LEI ---
--       while_loop.lua:34: LEI
--       while_loop.lua:35: GETTABUP
--       while_loop.lua:35: GETFIELD
--       while_loop.lua:35: SELF
--       while_loop.lua:35: GETTABUP
--       while_loop.lua:35: MOVE
--       while_loop.lua:35: CALL
--       while_loop.lua:35: LOADK
--       while_loop.lua:35: CALL
--       while_loop.lua:36: ADDI
--       while_loop.lua:36: JMP
--     --- End debugstrs ---
--     4
--     yk-execution: enter-jit-code: while_loop.lua:34: LEI
--     5
--     6
--     yk-execution: deoptimise ...
--     exit

local x = 0
while x <= 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")
