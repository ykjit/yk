local function make()
  return function(next)
    if next then
      next(nil)
    end
  end
end

local a = make()
local b = make()

-- Although 'a' and 'b' share a `Proto`, they are distinct `LClosure`s and
-- should not cause a recursive function to be traced.
a(b)
io.stderr:write("exit\n")

-- Run-time:
--   env-var: YK_HOT_THRESHOLD=0
--   env-var: YKD_LOG=4
--   env-var: YKD_SERIALISE_COMPILATION=1
--   stderr:
--     exit
