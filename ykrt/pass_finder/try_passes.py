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
from dataclasses import dataclass, field
from typing import List, Optional
from multiprocessing import Manager, Process, Queue, Value
import cargo_run
import setup_genetic
from copy import deepcopy 
import multiprocessing

RED = '\033[91m'
GREEN = '\033[92m'
PURPLE = '\033[38;5;128m'
YELLOW = '\033[93m'
RESET = '\033[0m'
maxlen = 0
# Stages in an LTO pipeline where optimisation passes can happen.
STAGES = "pre_link", "link_time"

@dataclass
class Pass:
    name: str
    on: int
    parent: Optional['Pass'] = None
    subpasses: 'Passes' = field(default_factory=lambda: Passes())

    def add_subpass(self, subpass_name: str):
        subpass = Pass(subpass_name, 1, self)
        self.subpasses.passes.append(subpass)

@dataclass
class Passes:
    passes: List[Pass] = field(default_factory=list)

    def add_pass(self, pass_obj: Pass):
        # print(f"{PURPLE}appending: {pass_obj}{RESET}")
        self.passes.append(pass_obj)

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
        return "opt-16 -passes='lto<O2>' -print-pipeline-passes < /dev/null 2>/dev/null"
    else:
        return "opt-16 -passes='lto-pre-link<O2>' -print-pipeline-passes < /dev/null 2>/dev/null" 

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

    return parts

def encode_passes_randomly(root_passes):
    def encode(passes):
        for i in range(len(passes.passes)):
            passes.passes[i].on = random.randint(0, 1)
            passes.passes[i].subpasses = encode(passes.passes[i].subpasses)
            subpasses = passes.passes[i].subpasses.passes
            if passes.passes[i].on == 1 and subpasses:
                subpass_inclusions = [subpass.on for subpass in subpasses]
                if all(inclusion == 0 for inclusion in subpass_inclusions):
                    idx = random.randint(0, len(subpasses) - 1)
                    passes.passes[i].subpasses.passes[idx].on = 1
        return passes
    root_passes_ = deepcopy(root_passes)
    root_passes_ = encode(root_passes_)
    return root_passes_

def decode_passes_to_string(passes):
    p_str = ''
    p_list = []
    for p in passes.passes:
        if p.on == 1:
            tmp = ''
            if p.subpasses.passes:
                tmp = decode_passes_to_string(p.subpasses)
                tmp = '(' + tmp + ')'
            tmp = p.name + tmp
            p_list.append(tmp)
    p_str = ','.join(p_list)
    return p_str

def parse_passes(s):
    root_passes = Passes()
    stack = [(root_passes, None)] 
    i = 0

    while i < len(s):
        start = i
        while i < len(s) and s[i] not in [',', '(', ')']:
            i += 1

        pass_name = s[start:i].strip()
        if pass_name and pass_name != "BitcodeWriterPass":
            current_pass = Pass(pass_name, 1, stack[-1][1])
            stack[-1][0].add_pass(current_pass)

        if i < len(s) and s[i] == '(':
            if pass_name and pass_name != "BitcodeWriterPass":
                stack.append((current_pass.subpasses, current_pass))
            i += 1
        elif i < len(s) and s[i] == ')':
            stack.pop()
            i += 1
        elif i < len(s) and s[i] == ',':
            i += 1
    return root_passes


def get_all_passes(is_prelink):
    """
    This function uses llvm's opt command to get list of prelink
    and postlink passes.
    """
    cmd = get_opt_cmd(is_prelink)
    sout = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert (sout.returncode != 0), "Opt command failed."
    sout = sout.stdout.decode('utf-8')
    pass_descrs = parse_passes(sout)
    passes = []
    seen = set()
    split_pass_str = split_passes(sout)
    for descr in split_pass_str:
        if descr not in seen:
            seen.add(descr)
            if descr != "BitcodeWriterPass":
                passes.append(Pass(descr, 1))
    return pass_descrs, passes


def test_pipeline(glogf, pl, id, yk_path, yklua_path):
    """
    The function sets PRELINK and POSTLINK env variable
    and then run test oracle.
    """
    global maxlen 
    sys.stdout.write(str(pl) + "...")
    sys.stdout.flush()

    log(glogf, "\n\n" + str(pl) + "\n")

    log(glogf, f"\nbefore assert")
    assert (len(pl.pre_link) != 0 or len(pl.link_time) != 0), "Both prelink and postlink passes cannot be empty!!!"
    log(glogf, f"\nafter assert ")

    env = os.environ.copy()  # Create a copy of the environment
    env[f"PRELINK_PASSES_{id}"] = ",".join([p.name for p in pl.pre_link])
    env[f"POSTLINK_PASSES_{id}"] = ",".join([p.name for p in pl.link_time])

    ret, time = cargo_run.run_test(id, yk_path, yklua_path, env=env) 

    if ret == 0:
        print(" [OK]")
        log(glogf, str(pl) + ": OK\n")
        return True, time
    else:
        log(glogf, str(pl) + " : FAILED\n")
        print(" [FAIL]") 
        return False, None

def list_of_passes_to_str(passes):
    return ",".join([str(p) for p in passes])

def evaluate_fitness(glogf, is_prelink, tasks, passes, tmpdir, id, fitness_scores, processing, run):
    """
    Returns fitness score based on whether the passes list 
    successfully builds the pipeline and runs all the tests.
    """ 
    log(glogf, f"\nid: {id}")
    yklua_path = os.path.join(tmpdir, 'yklua')
    yk_path = os.path.join(tmpdir, 'yk')

    curdir = os.getcwd()
    os.chdir(tmpdir) 
    
    while run.value == 1: 
        try:
            entity = tasks.get(block=False) 
        except queue.Empty:
            print(f"{RED}Closing Worker : Queue Empty{RESET}")
            break
        else:
            processing.value += 1
            try_passes = decode_passes_to_string(entity[1]) 
            pass_list = split_passes(try_passes)
            passes = []
            seen = set()

            for descr in pass_list:
                if descr not in seen:
                    seen.add(descr)
                    if descr != "BitcodeWriterPass":
                        passes.append(Pass(descr, 1))
            
            print(f"{PURPLE}Print passes: {passes}{RESET}")
            config = get_pipeline_config(is_prelink, passes)
            ret, exec_time = test_pipeline(glogf, config, id, yk_path, yklua_path)
            if ret:
                    exec_time = float(exec_time)
                    fitness_scores.append([entity[0], exec_time])
                    print(f"Adding {entity[0]}, {exec_time} to fitness_scores.")
            else:
                fitness_scores.append([entity[0], float('inf')])
                print(f"Adding {entity[0]}, {exec_time} to fitness_scores.")

            processing.value -= 1
            log(glogf, f"\nprocessing.value = {processing.value}")


    log(glogf, f"\nClosing Worker : Run False")
    glogf.close()
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
    # Assuming parent1 and parent2 are dictionaries like {'pass_name': '0 or '1', ...}
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)

    # Randomly select keys for crossover from the union of both parents' keys
    # all_keys = list(set(parent1.keys()).union(set(parent2.keys())))
    crossover_keys_num = random.randint(1, len(parent1.passes) - 1)
    crossover_indices = random.sample([i for i in range(0, len(parent1.passes))], crossover_keys_num)

    # Swap the Pass object at selected indices between parents for children
    for idx in crossover_indices:
        child1.passes[idx] = deepcopy(parent2.passes[idx])
        child2.passes[idx] = deepcopy(parent1.passes[idx])
    
    return child1, child2

def mutate(entity, mutation_rate):
    """
    Applies mutation to an entity by flipping each bit with a probability 
    defined by the mutation rate. This function supports binary-encoded entities.
    """
    entity_ = deepcopy(entity)
    for i in range(len(entity.passes)):
        if entity_.passes[i].subpasses.passes:
            entity_.passes[i].subpasses = mutate(entity_.passes[i].subpasses, mutation_rate)

        if random.random() < mutation_rate:
            entity_.passes[i].on = (1 - entity_.passes[i].on)
    
    return entity_

def genetic_algorithm(glogf, parsed_str, is_prelink, population_size, mutation_rate,
                      generations, target_fitness, passes, tmpdirs, ncpu):
    """
    Executes a genetic algorithm for finding optimization passes. It randomly generates 
    a population, evolves it through crossover and mutation, and selects entities based on 
    the fitness scores (which depends on the benchmark's execution time).
    """
    population = []
    fitness = []

    print(f"{GREEN}population size: {population_size} {RESET}")
    for i in range(population_size):
        entity = encode_passes_randomly(parsed_str)
        population.append(entity)
    
    print(f"{GREEN}len of population: {len(population)}{RESET}") 

    log(glogf, f"Population: {len(population)}\n") 
    for generation in range(generations):
        log(glogf, "=========================================================")
        log(glogf, f"\ngeneration: {generation}\n")

        with Manager() as manager:
            fitness_scores = manager.list()
            tasks = Queue()
            processes = []
            processing = Value("i", 0)
            run = Value("i", 1) # control worker termination 
            for i in range(len(population)):
                tasks.put([i, population[i]])
 
            loggers = []
            # Start workers
            n_processes = min(ncpu, population_size)
            print(f"{PURPLE}nprocess: {n_processes}{RESET}")
            for w in range(n_processes):
                # TODO: make tmpdir = tmpdirs[w/2] to assign 2 threads to every core (for bencher9) 
                tmpdir = tmpdirs[w]
                id = w
                genetic_log_path = f"/vol/extra/shreei/logs/genetic_multi_{w}.log"
                log_f = open(genetic_log_path, "w+")
                loggers.append(log_f)
                p = Process(target=evaluate_fitness, args=(glogf, is_prelink,
                                                           tasks, passes,
                                                           tmpdir, id, fitness_scores, processing, run))

                processes.append(p)
                # This command make sures single process runs on 1 core
                # TODO: change w to w/2 to pin 2 processes to 1 core on bencher9
                os.system(f"taskset -p -c {w} {p.pid}")   
                p.start()
                        
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
            
            print(f"{PURPLE}fitness: {fitness}{RESET}")
            log(glogf, f"\nfitness score: {fitness}\n")
            log(glogf, "=========================================================")
            
            if any(y <= target_fitness for y in fitness):
                print(f"Target fitness reached in generation {generation + 1}!")
                break

            elites = []
            for i in range(len(population) - 1):
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
                entity = deepcopy(population[0])
                entity = encode_passes_randomly(entity) 
                new_population.append(entity)

            population = new_population
    
    best_entity = population[fitness.index(min(fitness))]
    log(glogf, f"\nbest entity so far: {best_entity}\n")
    print(f"{PURPLE}{best_entity}{RESET}")
    return best_entity

def main(glogf, is_prelink, yk_path, yklua, cwd, base_temp_dir):
    
    # TODO: Make this remove -1 and double this for bencher9 
    num_cores = (multiprocessing.cpu_count() - 1)
    temp_directories = [] 

    curr_dir = os.getcwd()
    temp_directories = setup_genetic.setup(curr_dir, base_temp_dir, yk_path, yklua)
    # sys.exit()
    parsed_str, passes = get_all_passes(is_prelink)
    #FIXME: choose a better value for target fitness
    # currently choosing 0 secs, so the benchmark converges
    target_fitness = 0.0 
    best_entity = genetic_algorithm(glogf,
        parsed_str,
        is_prelink,
        population_size = 50, # len(passes) * 2,
        mutation_rate = 0.1,
        generations = 2,
        target_fitness = target_fitness,
        passes = passes,
        tmpdirs = temp_directories,
        ncpu = num_cores,
    )

    final_passes = decode_passes_to_string(best_entity) 
    log(glogf, f"\nFinal passes: {final_passes}\n")
    print(f"\n{PURPLE}{final_passes}{RESET}")

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
