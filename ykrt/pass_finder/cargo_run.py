import os
import subprocess
import time

def run_test(yk_path, env=None, n=1):
    # Set path and environment variables
    yk_path = os.environ.get('YK_PATH')
    if yk_path is None:
        raise ValueError("YK_PATH environment variable is not set")

    yklua_path = os.environ.get('YKLUA_PATH')
    if yklua_path is None:
        raise ValueError("YKLUA_PATH environment variable is not set")

    env = os.environ.copy()
    env['PATH'] = f"{yk_path}/bin:{env['PATH']}"
    env['YK_BUILD_TYPE'] = 'debug'
     
    # Check if pre and post-link flags are set
    prelink_passes = env.get('PRELINK_PASSES', 'Not Set')
    print(f"\n\033[93mPRELINK_PASSES: {prelink_passes}\033[0m")
    postlink_passes = env.get('POSTLINK_PASSES', 'Not Set')
    print(f"\n\033[93mPOSTLINK_PASSES: {postlink_passes}\033[0m")
 
    curdir = os.getcwd()
    os.chdir(yklua_path)

    times = []
    c_test = None 

    for _ in range(n):
        c = subprocess.run(["timeout 30 sh test.sh"], shell=True, env=env)
        print(f"\033[95m returncode for c is {c.returncode} \033[0m\n")
       
        os.chdir(yk_path)
        r = subprocess.run(["timeout 50 cargo test"], shell=True, env=env)

        if c.returncode == 0 and r.returncode == 0:
            os.chdir(yklua_path)
            before = time.time()
            c_test = subprocess.run(["timeout 8 sh run.sh"], shell=True, env=env)
            elapsed = time.time() - before
            times.append(elapsed)

    if len(times) != 0:
        mean_time = sum(times) / len(times)
    else:
        mean_time = 0

    if c_test is None:
        return 130, 0
    
    os.chdir(curdir)
    return c.returncode, mean_time


