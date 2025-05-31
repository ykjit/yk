-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YK_SIDETRACE_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     <0
--     <1
--     <2
--     yk-tracing: start-tracing: sidetrace.lua:59: LTI
--     <3
--     yk-tracing: stop-tracing: sidetrace.lua:59: LTI
--     --- Begin debugstrs: header: sidetrace.lua:59: LTI ---
--       sidetrace.lua:59: LTI
--       sidetrace.lua:60: GETTABUP
--       sidetrace.lua:60: GETFIELD
--       sidetrace.lua:60: SELF
--       sidetrace.lua:60: LOADK
--       sidetrace.lua:60: GETTABUP
--       sidetrace.lua:60: MOVE
--       sidetrace.lua:60: CALL
--       sidetrace.lua:60: LOADK
--       sidetrace.lua:60: CALL
--       sidetrace.lua:60: JMP
--       sidetrace.lua:58: FORLOOP
--     --- End debugstrs ---
--     <4
--     yk-execution: enter-jit-code: sidetrace.lua:59: LTI
--     yk-execution: deoptimise
--     >=5
--     yk-execution: enter-jit-code: sidetrace.lua:59: LTI
--     yk-execution: deoptimise
--     yk-tracing: start-side-tracing: sidetrace.lua:59: LTI
--     >=6
--     yk-tracing: stop-tracing: sidetrace.lua:59: LTI
--     --- Begin debugstrs: side-trace: sidetrace.lua:59: LTI ---
--       sidetrace.lua:62: GETTABUP
--       sidetrace.lua:62: GETFIELD
--       sidetrace.lua:62: SELF
--       sidetrace.lua:62: LOADK
--       sidetrace.lua:62: GETTABUP
--       sidetrace.lua:62: MOVE
--       sidetrace.lua:62: CALL
--       sidetrace.lua:62: LOADK
--       sidetrace.lua:62: CALL
--       sidetrace.lua:58: FORLOOP
--     --- End debugstrs ---
--     >=7
--     yk-execution: enter-jit-code: sidetrace.lua:59: LTI
--     >=8
--     >=9
--     >=10
--     yk-execution: deoptimise
--     exit

for i = 0, 10 do
  if i < 5 then
    io.stderr:write("<", tostring(i), "\n")
  else
    io.stderr:write(">=", tostring(i), "\n")
  end
end
io.stderr:write("exit\n")
