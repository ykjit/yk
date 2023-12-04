import os, subprocess, time

def run_test(yk_path,env=None, n=1) :
    if env is None:
        env = os.environ.copy()
    env['PATH'] = f"{yk_path}/bin:{env['PATH']}"
    env['YK_BUILD_TYPE'] = 'debug'
    prelink_passes = (env or os.environ).get('PRELINK_PASSES', 'Not Set')
    print(f"\033[93mPRELINK_PASSES: {prelink_passes}\033[0m")
    os.chdir(yk_path)
    r = subprocess.run(["timeout 50 cargo test"], shell=True, env=env or os.environ)
    # assert r.returncode == 0
    if r.returncode == 0:
        os.chdir("/home/shreei/research/yklua")
        times = []
        for _ in range(n):
            subprocess.run(["make clean"], shell=True, env=env)
            c = subprocess.run(["make && sh test.sh"], shell=True, env=env or os.environ)
            # assert c.returncode == 0
            if c.returncode == 0:
                before = time.time()
                c_test = subprocess.run(["sh run.sh"], shell=True, env=env or os.environ)
                elapsed = time.time() - before
                times.append(elapsed)
        # assert c_test.returncode == 0
        mean_time = sum(times) / len(times)
        os.chdir(yk_path)
        return c_test.returncode, mean_time
    os.chdir(yk_path)
    return r.returncode, 0 
    # print(mean_time)
    # print(f"returncode: {c.returncode}") 

code, time_p = run_test("/home/shreei/research/yk_pv")
print(code, " ",  time_p)


