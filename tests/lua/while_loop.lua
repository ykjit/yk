local x = 0
while x <= 6 do
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
--     yk-tracing: start-tracing: while_loop.lua:2: LEI
--     3
--     yk-tracing: stop-tracing: while_loop.lua:2: LEI
--     --- Begin debugstrs: while_loop.lua:2: LEI ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     while_loop.lua:2: LEI
--     while_loop.lua:3: GETTABUP
--     while_loop.lua:3: GETFIELD
--     while_loop.lua:3: SELF
--     while_loop.lua:3: GETTABUP
--     while_loop.lua:3: MOVE
--     while_loop.lua:3: CALL
--     while_loop.lua:3: LOADK
--     while_loop.lua:3: CALL
--     while_loop.lua:4: ADDI
--     while_loop.lua:4: JMP
--     --- End debugstrs ---
--     4
--     yk-execution: enter-jit-code: while_loop.lua:2: LEI
--     5
--     6
--     yk-execution: deoptimise TraceId(0) GuardId(33)
--     exit
