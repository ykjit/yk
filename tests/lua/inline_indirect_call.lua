-- ignore-if: test "$YKB_TRACER" = "swt"
-- Run-time:
--   env-var: YK_HOT_THRESHOLD=3
--   env-var: YKD_LOG=4
--   env-var: YKD_LOG_IR=jit-pre-opt
--   env-var: YKD_SERIALISE_COMPILATION=1

-- FIXME: I don't think there's any output we could easily check to ensure
-- luaB_assert has been inlined?

for i = 0, 100 do
    A = {}
    assert(A) -- causes an icall in the trace.
end

io.stderr:write("exit\n")
