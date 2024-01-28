#!/usr/bin/env python3
"""
usage: python3 try_passes.py <-lto -prelink] -- path path/to/test/oracle

This script aims to identify an LLVM AOT pipeline that satisfies a test oracle.
A typical test oracle would be a script that builds an interpreter and runs a
test suite. This script is required because some passes don't yet work with Yk.

The user specifies at least one pipeline (i.e. -prelink or -lto). The algorithm
then starts with the default clang -O2 pipeline which is sent to the test
oracle for evaluation. Each time the oracle is invoked it is passed a list of
prelink and postlink passes in the environment variable (PRELINK_PASSES and
POSTLINK_PASSES). The oracle must ensure that these pipelines are used when
compiling the interpreter (usually by passing the pipeline to the
--pre-link-pipeline and --post-link-pipeline arguments of yk-config). If the
oracle accepts the pipeline (it returns zero), then the passes are appended to
an "accept list". If the oracle rejects the pipeline (it returns non-zero),
then the algorithm divides the failing pipeline into two parts and tests the
two halves independently (in parallel). The search continues recursively until
the search space is exhausted. Upon termination the final accept list is
printed.
"""
import argparse, os, random, shutil, subprocess, sys, time, queue 
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue, Value
import cargo_run
import setup_genetic
import multiprocessing

RED = '\033[91m'
GREEN = '\033[92m'
PURPLE = '\033[38;5;128m'
RESET = '\033[0m'

# Stages in an LTO pipeline where optimisation passes can happen.
STAGES = "pre_link", "link_time"
@dataclass
class Pass:
    name: str
    def __str__(self):
        return self.name

@dataclass
class PipelineConfig:
    pre_link: list
    link_time: list

    def __str__(self):
        pre_link = ",".join([ str(p) for p in self.pre_link ])
        link_time = ",".join([ str(p) for p in self.link_time ])
        return f"PipelineConfig(pre_link=[{pre_link}], link_time=[{link_time}])"

def get_pipeline_config(is_prelink, try_passes, ok_passes=None):
    """Returns pipeline configuration based on the flag type."""
    
    if ok_passes is None:
        ok_passes = [] 
    if not is_prelink:
        return PipelineConfig([], ok_passes + try_passes)
    else:
        return PipelineConfig(ok_passes + try_passes, [])

def get_opt_cmd(is_prelink):
    """Returns command based on the flag type."""
    
    if not is_prelink:
        return "opt -passes='lto<O2>' -print-pipeline-passes < /dev/null 2>/dev/null"
    else:
        return "opt -passes='lto-pre-link<O2>' -print-pipeline-passes < /dev/null 2>/dev/null" 

def log(logf, s):
    logf.write(s)
    logf.flush()

def executable_exists(cmd):
    return shutil.which(cmd) is not None

def split_passes(passes_string):
    """
    Split comma-separated passes list into individual passes.
    Uses a single counter for both parentheses and angle brackets.
    """
    parts = []
    temp_part = []
    nesting_level = 0

    for c in passes_string:
        if c == '(' or c == '<':
            nesting_level += 1
        elif c == ')' or c == '>':
            nesting_level -= 1

        if c == ',' and nesting_level == 0:
            part = ''.join(temp_part).strip()
            if part:
                parts.append(part)
            temp_part = []
        else:
            temp_part.append(c)
 
    # Add the last part if it's not empty
    part = ''.join(temp_part).strip()
    if part:
        parts.append(part)
    print(f"{RED}length: {len(parts)}{RESET}")    
    print(f"{GREEN}passes{parts}{RESET}\n")
    return parts

def get_all_passes(is_prelink):
    """
    This function uses llvm's opt command to get list of prelink
    and postlink passes.
    """
    cmd = get_opt_cmd(is_prelink)
    sout = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert (sout.returncode != 0), "Opt command failed."
    sout = sout.stdout.decode('utf-8')
    pass_descrs = split_passes(sout)
    passes = []
    seen = set()

    for descr in pass_descrs:
        if descr not in seen:
            seen.add(descr)
            if descr != "BitcodeWriterPass":
                passes.append(Pass(descr))
      
    print(f"Found {len(passes)} passes")    
    return passes

def test_pipeline(logf, pl, id, yk_path, yklua_path):
    """
    The function sets PRELINK and POSTLINK env variable
    and then run test oracle.
    """
    sys.stdout.write(str(pl) + "...")
    sys.stdout.flush()

    log(logf, "\n\n" + str(pl) + "\n")

    # Make sure we don't run empty strings in pipeline.
    assert (len(pl.pre_link) != 0 or len(pl.link_time) != 0), "Both prelink and postlink passes cannot be empty!!!"

    env = os.environ.copy()  # Create a copy of the environment
    env[f"PRELINK_PASSES_{id}"] = ",".join([p.name for p in pl.pre_link]) 
    env[f"POSTLINK_PASSES_{id}"] = ",".join([p.name for p in pl.link_time])

    ret, time = cargo_run.run_test(id, yk_path, yklua_path, env=env)
    print(f"{PURPLE}ret code for cargo run is {ret}{RESET}")
    if ret == 0:
        print(" [OK]")
        log(logf, str(pl) + ": OK\n")
        return True, time
    else:
        log(logf, str(pl) + " : FAILED\n")
        print(" [FAIL]") 
        return False, None

def list_of_passes_to_str(passes):
    return ",".join([str(p) for p in passes])

def evaluate_fitness(glogf, is_prelink, tasks, passes, tmpdir, id, fitness_scores, processing, run):
    """
    Returns fitness score based on whether the passes list 
    successfully builds the pipeline and runs all the tests.
    """ 
    yklua_path = os.path.join(tmpdir, 'yklua')
    yk_path = os.path.join(tmpdir, 'yk')

    curdir = os.getcwd()
    os.chdir(tmpdir) 

    while run.value == 1:
        print(f"{PURPLE}in while{RESET}")
        try:
            print(f"{PURPLE}in try{RESET}")
            entity = tasks.get(block=False) 
        except queue.Empty:
            print(f"{PURPLE}in except{RESET}")
            print(f"{RED}Closing Worker : Queue Empty{RESET}")
            break
        else:
            processing.value += 1
            print(f"{PURPLE}in else{RESET}")
            try_passes = [passes[i] for (i, bit) in enumerate(entity[1]) if bit]
            config = get_pipeline_config(is_prelink, try_passes)
            ret, exec_time = test_pipeline(glogf, config, id, yk_path, yklua_path) 
            if ret:
                    exec_time = float(exec_time)
                    print(f"{PURPLE}exec time: {exec_time}{RESET}")
                    fitness_scores.append([entity[0], exec_time])
            else:
                # Return execution time as a large positive value
                print(f"{PURPLE}exec time: INF {RESET}")
                fitness_scores.append([entity[0], float('inf')])
            processing.value -= 1
    print(f"{RED}Closing Worker : Run False{RESET}")
    os.chdir(curdir)
    return True

def tournament(population, fitness, sp):
    """
    Selects a parent from the given population using the tournament selection method.

    This function implements tournament selection for genetic algorithms.
    Two individuals are randomly selected from the population. The one with the better fitness 
    (lower fitness value) is chosen as the parent with a probability of `sp` (selection probability).
    If not chosen based on `sp`, the other individual is selected as the parent.
    """
    population_ = [(i, population[i]) for i in range(len(population))]
    parent1 = random.choices(population_, weights=[1]*len(population), k=1)[0]
    parent2 = random.choices(population_, weights=[1]*len(population), k=1)[0]
   
    while parent1 == parent2:
        parent2 = random.choices(population_, weights=[1]*len(population), k=1)[0]
   
    if random.random() > sp:
        parent = parent1[1] if fitness[parent1[0]] < fitness[parent2[0]] else parent2[1]
    else:
        parent = parent2[1] if fitness[parent1[0]] < fitness[parent2[0]] else parent1[1]

    return parent

def crossover(parent1, parent2):
    """
    Performs crossover between two parents using a single crossover point.
    The crossover point is randomly chosen, and two children are created
    by combining segments of the parents. Parent selection is based on the
    roulette method.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(entity, mutation_rate):
    """
    Applies mutation to an entity by flipping each bit with a probability 
    defined by the mutation rate. This function supports binary-encoded entities.
    """
    mutated_entity = []
    for bit in entity:
        if random.random() < mutation_rate:
            mutated_entity.append(1)
        else:
            mutated_entity.append(bit)
    return mutated_entity

def genetic_algorithm(glogf, is_prelink, population_size, mutation_rate,
                      generations, target_fitness, passes, tmpdirs, ncpu):
    """
    Executes a genetic algorithm for finding optimization passes. It randomly generates 
    a population, evolves it through crossover and mutation, and selects entities based on 
    the fitness scores (which depends on the benchmark's execution time).
    """
    population = []
    fitness = []

    for _ in range(population_size):
        entity = [random.randint(0,1) for _ in range(len(passes))]
        population.append(entity)

    for generation in range(generations):
        log(glogf, "=========================================================")
        log(glogf, f"\ngeneration: {generation}\n")

        with Manager() as manager:
            fitness_scores = manager.list()
            tasks = Queue()
            processes = []
            processing = Value("i", 0)
            run = Value("i", 1) # control worker termination 
            # Prepare tasks out of population. Task = [entity_id, entity].
            for i in range(len(population)):
                tasks.put([i, population[i]])
 
            # Start workers
            for w in range(ncpu):
                # TODO: make tmpdir = tmpdirs[w/2] to assign 2 threads to every core (for bencher9)
                print(f"{PURPLE}w is {w}{RESET}")
                tmpdir = tmpdirs[w]
                id = w
                p = Process(target=evaluate_fitness, args=(glogf, is_prelink,
                                                           tasks, passes,
                                                           tmpdir, id, fitness_scores, processing, run))
                print(f"{RED}process: {p}{RESET}")
                processes.append(p)
                # This command make sures single process runs on 1 core
                os.system(f"taskset -p -c {w} {p.pid}")   # TODO: change w to w/2 to pin 2 processes to 1 core on bencher9
                p.start()
            
            print(f"{GREEN}processes list: {processes}{RESET}")
            
            # Time to let at least one thread to start processing
            time.sleep(5)
            while processing.value > 0:
                time.sleep(5)
                print(f"{GREEN}processing.value: {processing.value}{RESET}")
            
            # close workers
            run.value = 0
            
            fitness_scores.sort() # sort by entity id
            print(f"{PURPLE}fitness_scores: {fitness_scores}{RESET}")
            fitness = [score[1] for score in fitness_scores]

            log(glogf, f"\nfitness score: {fitness}\n")
            log(glogf, "=========================================================")
            
            if any(y <= target_fitness for y in fitness):
                print(f"Target fitness reached in generation {generation + 1}!")
                break

            elites = []
            for i in range(len(population)):
                if fitness[i] < float('inf'):
                    elites.append(population[i])

            parents = []
            sp = 0.1 
            
            for i in range((len(population) - len(elites)) // 2):
                parent1 = tournament(population, fitness, sp)
                parent2 = tournament(population, fitness, sp)
                parents.append((parent1, parent2))
             
            # Perform crossover and mutation to create a new generation
            new_population = []
            new_population.extend(elites)
            for parent1, parent2 in parents:
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                new_population.extend([child1, child2])
            
            padding = len(population) - len(new_population)
            for i in range(padding):
                entity = [random.randint(0,1) for _ in range(len(passes))]
                new_population.append(entity)

            population = new_population
    
    best_entity = population[fitness.index(min(fitness))]
    print(f"{PURPLE}{best_entity}{RESET}")
    return best_entity  

def main(glogf, is_prelink, yk_path, yklua, cwd, base_temp_dir):
    # Sanity check, test script should work with no extra passes.
    # assert(test_pipeline(logf, PipelineConfig([], [])))
    
    # TODO: Make this remove -1 and double this for bencher9 
    num_cores = multiprocessing.cpu_count() - 1
    temp_directories = [] 

    curr_dir = os.getcwd()
    temp_directories = setup_genetic.setup(curr_dir, base_temp_dir, yk_path, yklua)
    passes = get_all_passes(is_prelink)

    #FIXME: choose a better value for target fitness
    # currently choosing 0 secs, so the benchmark converges
    target_fitness = 0.0 
    best_entity = genetic_algorithm(glogf,
        is_prelink,
        population_size = len(passes) * 2,
        mutation_rate = 0.1,
        generations = 1,
        target_fitness = target_fitness,
        passes = passes,
        tmpdirs = temp_directories,
        ncpu = num_cores,
    )

    final_passes = [passes[i].name for (i, bit) in enumerate(best_entity) if bit]  
    log(glogf, f"\nFinal passes: {final_passes}\n")
    print(f"{PURPLE}{final_passes}{RESET}")

if __name__ == "__main__":
    if not os.environ.get('YK_PATH') or not os.environ.get('YKLUA_PATH'):
        print("Please set both YK_PATH and YKLUA_PATH environment variables before running the script.")
        exit(1)
    
    if '--help' in sys.argv:
        print(__doc__)
        exit(0)

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-lto', action='store_true', help='Set flag for LTO.')
    group.add_argument('-prelink', action='store_true', help='Set flag for Prelink.')
    parser.add_argument("base_dir", type=str)
    args = parser.parse_args()

    is_prelink = args.prelink
    base_dir = args.base_dir

    if not is_prelink and not args.lto:
        print("Flag invalid! Please provide a valid flag: -lto or -prelink")
        exit(1) 

    # Set the global variable with the parsed path
    yk_path = os.environ.get('YK_PATH')
    if yk_path is None:
        raise ValueError("YK_PATH environment variable is not set")

    yklua_path = os.environ.get('YKLUA_PATH')
    if yklua_path is None:
        raise ValueError("YKLUA_PATH environment variable is not set")

    CWD = yk_path
    genetic_log_path = os.path.join(CWD, "ykrt/pass_finder/genetic.log")
    print(f"PATH to interpreter: {CWD}")

    with open(genetic_log_path, "w+") as glogf:
            main(glogf, is_prelink, yk_path, yklua_path, CWD, base_dir)
