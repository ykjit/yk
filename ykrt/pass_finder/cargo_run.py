import os
import subprocess
import time

RED = '\033[91m'
GREEN = '\033[92m'
PURPLE = '\033[38;5;128m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def run_test(id, yk_path, yklua_path, env, n=1):
    """
    The function loads the environment variables and
    runs YK's test and YKLUA build plus tests. If both succeeds
    the function then runs the benchmark script to get
    the execution time which is used by the genetic algorithm
    for fitness function.
    """
    #TODO: directly send the path instead of checking env variable
    if yk_path is None:
        raise ValueError("YK_PATH environment variable is not set") 
    if yklua_path is None:
        raise ValueError("YKLUA_PATH environment variable is not set") 
    # Check if pre and post-link flags are set

    prelink_passes = env.get(f'PRELINK_PASSES_{id}', 'Not Set') 
    print(f"\n{YELLOW}PRELINK_PASSES: {prelink_passes}{RESET}")
    postlink_passes = env.get(f'POSTLINK_PASSES_{id}', 'Not Set')
    print(f"\n{YELLOW}POSTLINK_PASSES: {postlink_passes}{RESET}")
  
    curdir = os.getcwd()
    os.chdir(yklua_path)
    times = []
    c_test = None 
    for _ in range(n):
        subprocess.run(["make clean"], shell=True, env=env)
        c = subprocess.run(["make && timeout 30 sh test.sh"], shell=True, env=env or os.environ)
        os.chdir(yk_path)
        if c.returncode == 0:
            r = subprocess.run(f"timeout 60 cargo test --test c_tests", shell=True, env=env or os.environ)
        else:
            break
        os.chdir(yklua_path)
        os.chdir(os.path.join(yklua_path, 'tests'))
        lua_interpreter_path = os.path.join(yklua_path, 'src', 'lua')
        cmd = f"timeout 5 {lua_interpreter_path} db.lua"
        if c.returncode == 0 and r.returncode == 0:
            before = time.time()
            c_test = subprocess.run(cmd, shell=True, env=env or os.environ)
            elapsed = time.time() - before
            times.append(elapsed)

    if len(times) != 0:
        mean_time = sum(times) / len(times)
    else:
        mean_time = None

    if c_test is None:
        return 130, None
    elif c_test.returncode != 0:
        return c_test.returncode, None

    os.chdir(curdir)
    return 0, mean_time
