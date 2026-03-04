function f(x)
  io.stderr:write(tostring(x), "\n")
  if x > 0 then
    f(x - 1)
  end
end

f(6)
io.stderr:write("exit\n")

-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     6
--     5
--     4
--     yk-tracing: start-tracing: recursive_function.lua:2: GETTABUP
--     3
--     yk-tracing: stop-tracing: recursive_function.lua:2: GETTABUP
--     --- Begin debugstrs: recursive_function.lua:2: GETTABUP ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     recursive_function.lua:2: GETTABUP
--     recursive_function.lua:2: GETFIELD
--     recursive_function.lua:2: SELF
--     recursive_function.lua:2: GETTABUP
--     recursive_function.lua:2: MOVE
--     recursive_function.lua:2: CALL
--     recursive_function.lua:2: LOADK
--     recursive_function.lua:2: CALL
--     recursive_function.lua:3: GTI
--     recursive_function.lua:4: GETTABUP
--     recursive_function.lua:4: ADDI
--     recursive_function.lua:4: CALL
--     --- End debugstrs ---
--     2
--     yk-execution: enter-jit-code: recursive_function.lua:2: GETTABUP
--     1
--     0
--     yk-execution: deoptimise ...
--     exit
