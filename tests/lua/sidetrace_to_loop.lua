-- Run-time:
--   env-var: YK_HOT_THRESHOLD=2
--   env-var: YK_SIDETRACE_THRESHOLD=2
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=debugstrs
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     h1
--     yk-tracing: start-tracing: sidetrace_to_loop.lua:78: GTI
--     yk-tracing: stop-tracing: sidetrace_to_loop.lua:78: GTI
--     --- Begin debugstrs: sidetrace_to_loop.lua:78: GTI ---
--     ; {
--     ;   "trid": "0",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     sidetrace_to_loop.lua:78: GTI
--     sidetrace_to_loop.lua:79: TEST
--     sidetrace_to_loop.lua:79: JMP
--     sidetrace_to_loop.lua:88: ADDI
--     sidetrace_to_loop.lua:88: JMP
--     --- End debugstrs ---
--     h2
--     yk-execution: enter-jit-code: sidetrace_to_loop.lua:78: GTI
--     yk-execution: deoptimise ...
--     yk-execution: enter-jit-code: sidetrace_to_loop.lua:78: GTI
--     yk-execution: deoptimise ...
--     h3
--     yk-execution: enter-jit-code: sidetrace_to_loop.lua:78: GTI
--     yk-execution: deoptimise ...
--     yk-tracing: start-side-tracing: sidetrace_to_loop.lua:78: GTI
--     yk-tracing: stop-tracing: sidetrace_to_loop.lua:83: FORLOOP
--     yk-tracing: start-tracing: sidetrace_to_loop.lua:83: FORLOOP
--     yk-tracing: stop-tracing: sidetrace_to_loop.lua:83: FORLOOP
--     --- Begin debugstrs: sidetrace_to_loop.lua:83: FORLOOP ---
--     ; {
--     ;   "trid": "2",
--     ;   "start": {
--     ;     "kind": "ControlPoint"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Loop"
--     ;   }
--     ; }
--     sidetrace_to_loop.lua:83: FORLOOP
--     --- End debugstrs ---
--     --- Begin debugstrs: sidetrace_to_loop.lua:78: GTI ---
--     ; {
--     ;   "trid": "1",
--     ;   "start": {
--     ;     "kind": "Guard",
--     ;     "src_trid": "0",
--     ;     "gidx": "${{5}}"
--     ;   },
--     ;   "end": {
--     ;     "kind": "Coupler",
--     ;     "tgt_trid": "2"
--     ;   }
--     ; }
--     sidetrace_to_loop.lua:82: TEST
--     sidetrace_to_loop.lua:83: LOADI
--     sidetrace_to_loop.lua:83: LOADI
--     sidetrace_to_loop.lua:83: LOADI
--     sidetrace_to_loop.lua:83: FORPREP
--     --- End debugstrs ---
--     yk-execution: enter-jit-code: sidetrace_to_loop.lua:83: FORLOOP
--     yk-execution: deoptimise ...
--     yk-execution: enter-jit-code: sidetrace_to_loop.lua:78: GTI
--     yk-execution: deoptimise ...
--     yk-tracing: start-side-tracing: sidetrace_to_loop.lua:78: GTI
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
