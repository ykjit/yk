local x = 0
for _ = 0, 6 do
  io.stderr:write(tostring(x), "\n")
  x = x + 1
end
io.stderr:write("exit\n")

-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     0
--     1
--     2
--     yk-tracing: start-tracing: for_loop.lua:3: GETTABUP
--     3
--     yk-tracing: stop-tracing: for_loop.lua:3: GETTABUP
--     --- Begin debugstrs: for_loop.lua:3: GETTABUP ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     for_loop.lua:3: GETTABUP
--     for_loop.lua:3: GETFIELD
--     for_loop.lua:3: SELF
--     for_loop.lua:3: GETTABUP
--     for_loop.lua:3: MOVE
--     for_loop.lua:3: CALL
--     for_loop.lua:3: LOADK
--     for_loop.lua:3: CALL
--     for_loop.lua:4: ADDI
--     for_loop.lua:2: FORLOOP
--     --- End debugstrs ---
--     4
--     yk-execution: enter-jit-code: for_loop.lua:3: GETTABUP
--     5
--     6
--     yk-execution: deoptimise ...
--     exit
