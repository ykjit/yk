-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     0
--     1
--     2
--     yk-tracing: start-tracing: for_loop.lua:43: GETTABUP
--     3
--     yk-tracing: stop-tracing: for_loop.lua:43: GETTABUP
--     --- Begin debugstrs: for_loop.lua:43: GETTABUP ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     for_loop.lua:43: GETTABUP
--     for_loop.lua:43: GETFIELD
--     for_loop.lua:43: SELF
--     for_loop.lua:43: GETTABUP
--     for_loop.lua:43: MOVE
--     for_loop.lua:43: CALL
--     for_loop.lua:43: LOADK
--     for_loop.lua:43: CALL
--     for_loop.lua:44: ADDI
--     for_loop.lua:42: FORLOOP
--     --- End debugstrs ---
--     4
--     yk-execution: enter-jit-code: for_loop.lua:43: GETTABUP
--     5
--     6
--     yk-execution: deoptimise ...
--     exit

local x = 0
for _ = 0, 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")
