import os, shutil, subprocess, sys, queue
import multiprocessing
from  multiprocessing import Manager, Process, Queue

def setup_worker(curr_dir, temp_directories, base_temp_dir, yk_path, yklua, tasks):
    while True:    
        try:
            i = tasks.get(block=False)
        except queue.Empty:
            print(f"Closing setup worker")
            break
        else:
            # Create a directory with the custom name
            temp_dir_name = f"tmp_{i}"
            temp_dir = os.path.join(base_temp_dir, temp_dir_name)
            git_repo_path = os.path.join(temp_dir, "yk")
            yklua_dest_path = os.path.join(temp_dir, "yklua")
            yk_test_src = os.path.join(git_repo_path, "tests", "src", "lib.rs")
            langtester = os.path.join(temp_dir, "lang_tester")
            os.makedirs(temp_dir, exist_ok=True)
            
            if not os.path.exists(langtester):
                shutil.copytree("/home/shreei/research/lang_tester", langtester)
            if not os.path.exists(git_repo_path):
                shutil.copytree(yk_path, git_repo_path, ignore=shutil.ignore_patterns('target'))
                os.chdir(git_repo_path)
                
                if os.path.exists(yk_test_src):
                    subprocess.run(f"sed -i -e 's/PRELINK_PASSES/PRELINK_PASSES_{i}/g' {yk_test_src}", shell=True)
                    subprocess.run(f"sed -i -e 's/POSTLINK_PASSES/POSTLINK_PASSES_{i}/g' {yk_test_src}", shell=True)
                #subprocess.run("git submodule init", shell=True, env=os.environ)
                #subprocess.run("git submodule update", shell=True, env=os.environ)
                subprocess.run("cargo test", shell=True, env=os.environ)
            elif os.path.exists(git_repo_path):
                os.chdir(git_repo_path)
                # subprocess.run("cargo test --test c_tests", shell=True, env=os.environ)
                print("..")
            else:
                print(f"Directory {git_repo_path} does not exist.")
                sys.exit()

            if not os.path.exists(yklua_dest_path):
                shutil.copytree(yklua, yklua_dest_path)   
                yklua_src = os.path.join(yklua_dest_path, "src")
                os.chdir(yklua_src)
                subprocess.run(f"sed -i -e 's/PRELINK_PASSES/PRELINK_PASSES_{i}/g' Makefile", shell=True) 
                subprocess.run(f"sed -i -e 's/POSTLINK_PASSES/POSTLINK_PASSES_{i}/g' Makefile", shell=True)  

            temp_directories.append(temp_dir)
            os.chdir(curr_dir)

def setup(curr_dir, base_temp_dir, yk_path, yklua):
    num_cores = multiprocessing.cpu_count() - 1
    directories = []

    with Manager() as manager:
        temp_directories = manager.list()
        tasks = Queue()
        processes = []
        
        #TODO: on bencher9 change num_cores to num_cores * 2
        for i in range(num_cores):
            tasks.put(i)

        for i in range(num_cores): 
            p = Process(target=setup_worker, args=(curr_dir, temp_directories, base_temp_dir, yk_path, yklua, tasks))
            processes.append(p)
            os.system(f"taskset -p -c {i} {p.pid}")
            p.start()

        for p in processes:
            p.join()

        directories = [dir for dir in temp_directories]

    return directories 
