import time
import tracemalloc
import numpy as np
import pandas as pd
import gc
import os
import psutil
import csv
import torch
import model.model as nn_arch
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

device = 'cuda'
input_sizes = [1, 10, 100, 1000]
seed_quantity = [10, 10, 10, 10]
output_file = f'iterative_vs_NN_comparison_{device}.csv'
nn_path = "/nn_data/both_PudgeSixLayer_2048/1106_002520/model_best.pth"


𝛿 = 0.08 #depreciation
β = 0.95 #discount factor

def load_consumption_model(nn_path,device):
    '''Load a model from a particular .pth file and assemble using structure contained in nn_arch.py'''
    #print("entered load_consumption_model")
    nn_path=f'{os.getcwd()}{nn_path}'
    modelinfo = torch.load(nn_path, map_location=torch.device(device))


    #print("loaded info")

    config = modelinfo['config']

    model = getattr(nn_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    #print("loaded state dict")


    if "cons_scale" in config["data_loader"]["args"]:
        cons_scale=config['data_loader']['args']['cons_scale']
    else:
        cons_scale=1    
    if "i_a_scale" in config["data_loader"]["args"]:
        i_a_scale=config['data_loader']['args']['i_a_scale']
    else:
        i_a_scale=1
    if "input_scale" in config["data_loader"]["args"]:
        input_scale=config['data_loader']['args']['input_scale']
    else:
        input_scale={}


    return model, cons_scale, i_a_scale, input_scale

def scale_input(input_data, scale_dict,inputID="input",verbose=True):
    '''
    Scales input data according to the input_scale dictionary provided
    '''
    if  scale_dict['dist'] in ["unif", "uniform"]:
        a,b=scale_dict['params']
        if verbose==True:
            print(f"Scaling requested for {inputID}, Distribution:",scale_dict['dist']," Parameters:", scale_dict['params'])
        input_data= (input_data - a)/b
        return input_data
    elif  scale_dict['dist'] in ["norm", "normal"]:
        mu, std = scale_dict['params']
        if verbose==True:
            print(f"Scaling requested for {inputID}, Distribution",scale_dict['dist']," Parameters:", scale_dict['params'])
        input_data = (input_data - mu)/std
        return input_data
    else:
        print(f"ERROR: Input scaling was not in recognized format. Neural network cannot be used as configured.")

def generate_input(size, seed=None):
    """Generate agent input data."""
    print(seed)
    np.random.seed(seed)
    variable_distributions = {
    "names": ["Alpha", "k", "Sigma", "Theta"],
    "num_vars": 4,
    "bounds": [[1.08, 0.074], [0.01, 40], [0.01, 2], [0.01, 1]],
    "dists": ["norm", "unif", "unif", "unif"]}
    samples = {}
    for i in range(variable_distributions["num_vars"]):
        name = variable_distributions["names"][i]
        dist = variable_distributions["dists"][i]
        bound = variable_distributions["bounds"][i]
        
        if dist == "norm":
            mean, std = bound
            samples[name] = np.random.normal(mean, std, size)
        elif dist == "unif":
            low, high = bound
            samples[name] = np.random.uniform(low, high, size)
     
    samples_df=pd.DataFrame(samples)
    samples_df["Sigma"]=np.round(samples_df["Sigma"],1)
    samples_df["Theta"]=np.round(samples_df["Theta"],1)
    samples_tensor = torch.tensor(samples_df.values, dtype=torch.float32).to(device)
    
    return samples_df, samples_tensor

def measure_performance(func, input_data):
    """Measure the performance of a function."""
    # Measure memory usage, execution time, GPU and CPU time
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.time()
    cpu_start_time = time.process_time()
    gpu_start_event, gpu_end_event = None, None
    if torch.cuda.is_available():
        gpu_start_event = torch.cuda.Event(enable_timing=True)
        gpu_end_event = torch.cuda.Event(enable_timing=True)
        gpu_start_event.record()
    result = func(input_data)
    if torch.cuda.is_available():
        gpu_end_event.record()
        torch.cuda.synchronize() 
    cpu_end_time = time.process_time()
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    execution_time = end_time - start_time
    cpu_time = cpu_end_time - cpu_start_time
    memory_usage = peak / 10**6  # MB
    gpu_memory_usage = np.nan
    gpu_time = np.nan
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_memory_usage = torch.cuda.max_memory_allocated() / 10**6  # MB
        gpu_time = gpu_start_event.elapsed_time(gpu_end_event) / 1000  # seconds


    # Measure garbage collection overhead
    gc_start_time = time.time()
    gc.collect()
    gc_end_time = time.time()
    gc_time = gc_end_time - gc_start_time

    # Measure I/O operations
    read_count = np.nan
    write_count = np.nan
    read_bytes =  np.nan
    write_bytes = np.nan
    try:
        process = psutil.Process(os.getpid())
        if hasattr(process, 'io_counters'):
            process = psutil.Process(os.getpid())
            io_counters = process.io_counters()
            read_count = io_counters.read_count
            write_count = io_counters.write_count
            read_bytes = io_counters.read_bytes
            write_bytes = io_counters.write_bytes
    except (AttributeError, psutil.AccessDenied):
        pass
    print ("returning from measure_performance")
    return {
        'execution_time': execution_time,
        'cpu_time': cpu_time,
        'memory_usage': memory_usage,
        'gpu_time': gpu_time,
        'gpu_memory_usage': gpu_memory_usage,
        'gc_time': gc_time,
        'read_count': read_count,
        'write_count': write_count,
        'read_bytes': read_bytes,
        'write_bytes': write_bytes,
    },result

def compare_functions(func_a, func_b, input_sizes, seed_quantity, output_file):
    metrics = []

    for size, num_seeds in zip(input_sizes, seed_quantity):
        print(f"Input size: {size}, Number of seeds: {num_seeds}")
        seeds=np.random.randint(0, 1000, num_seeds)
        print(seeds)
        for seed in seeds:
            print(f"Seed: {seed}")

            input_data_a,input_data_b = generate_input(size, seed)
            print(input_data_a)
            print(input_data_b)
            metrics_a,results_a = measure_performance(func_a, input_data_a)
            metrics_b,results_b = measure_performance(func_b, input_data_b)
            for i in range(size):
                incorrect_i_a=0
                if results_a["i_a"].iloc[i] != results_b["i_a"].iloc[i]:
                    incorrect_i_a+=1
            correct_i_a_percent=100-incorrect_i_a/size*100
            MAE=np.mean(np.abs(results_a["consumption"]-results_b["consumption"]))
            RMSE=np.sqrt(np.mean((results_a["consumption"]-results_b["consumption"])**2))
            MPE=np.mean((results_a["consumption"]-results_b["consumption"])/results_a["consumption"]*100)


            metrics.append([seed, size, 'Iterative', metrics_a['execution_time'], metrics_a['cpu_time'], metrics_a['memory_usage'], metrics_a['gpu_time'], metrics_a['gpu_memory_usage'], 
                            metrics_a['gc_time'], metrics_a['read_count'], metrics_a['write_count'], metrics_a['read_bytes'], metrics_a['write_bytes'],np.nan, np.nan, np.nan, np.nan, np.nan])
            metrics.append([seed, size, 'NN', metrics_b['execution_time'], metrics_b['cpu_time'], metrics_b['memory_usage'], metrics_b['gpu_time'], metrics_b['gpu_memory_usage'],
                            metrics_b['gc_time'], metrics_b['read_count'], metrics_b['write_count'], metrics_b['read_bytes'], metrics_b['write_bytes'],incorrect_i_a, correct_i_a_percent, MAE, RMSE, MPE])
            print(f"Seed {seed} completed")
            with open(output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Seed', 'Sample_Size', 'Function', 'Execution_Time_s', 'CPU_Time_s', 'Memory_Usage_MB','GPU_Time_s', 'GPU_Memory_Usage_MB', 
                                 'Garbage_C_Time_s', 'Read_Count', 'Write_Count', 'Read_Bytes', 'Write_Bytes', "Incorrect_i_a","Correct_i_a_Percent" "Consumption_MAE", "Consumption_RMSE", "Consumption_MPE"])
                writer.writerows(metrics)

    print(f"Results saved to {output_file}")

# Example functions for testing

TechTable = {#contains values for 0:gamma 1:cost 2:theta
    "low":   [0.3,  0   ],
    "medium":[0.35, 0.15],
    "high":  [0.45, 0.65]}

TechTableArray = np.array([[ 0.3,  0 ],[0.35, 0.15],[0.45, 0.65]])

AdapTable = {
    # contains values for 0:theta 1:cost 
    "none":   [  0, 0   ],
    "good":   [0.5, 0.20],
    "better": [0.5, 0.50]}

AdapTableArray = np.array([[ 0,  0 ],[0.5, 0.2],[0.9, 0.5]])

# Optimization Routine:


def maximize(g, a, b, args):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example 
    https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    Maximize the function g over the interval [a, b].

    The maximizer of g on any interval is
    also the minimizer of -g.  The tuple args collects any extra
    arguments to g.

    Returns the maximum value and the maximizer.
    """

    objective = lambda x: -g(x, *args)
    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum

def utility(c, σ, type="isoelastic"):
    if type == "isoelastic":
        if σ ==1:
            return np.log(c)
        else:
            return (c**(1-σ)-1)/(1-σ)

    else:
        print("Unspecified utility function!!!")


def income_function(k,α): 
    f = []
    for i in TechTable.keys(): 
        #in the end, they may need their own tech tables
        entry = α * k**TechTable[i][0] - TechTable[i][1]
        f.append(entry)
    return max(f)



class BellmanEquation:
     #Adapted from: https://python.quantecon.org/optgrowth.html
    def __init__(self,
                 u,            # utility function
                 f,            # production function
                 k,            # current state k_t
                 θ,            # given shock factor θ
                 σ,            # risk averseness
                 α,            # human capital
                 i_a,          # adaptation investment
                 m,            # protection multiplier
                 β=β,          # discount factor
                 𝛿=𝛿,          # depreciation factor 
                 name="BellmanNarrowExtended"):

        self.u, self.f, self.k, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m, self.name = u, f, k, β, θ, 𝛿, σ, α, i_a, m, name

        # Set up grid
        
        startgrid=np.array([1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1,2,3,4,5,6,7,8,9,10,k+100])

        ind=np.searchsorted(startgrid, k)
        self.grid=np.concatenate((startgrid[:ind],np.array([k*0.99995, k]),
                                 startgrid[ind:]))

        self.grid=self.grid[0<(np.array([income_function(x, self.α) for x in self.grid])+self.θ*(1-𝛿)*self.grid-self.i_a)*0.99995]

        # Identify target state k
        self.index = np.searchsorted(self.grid, k)-1
    
    def value(self, c, y, v_array):
        """
        Right hand side of the Bellman equation.
        """

        u, f, β, θ, 𝛿, σ, α, i_a, m = self.u, self.f, self.β, self.θ, self.𝛿, self.σ, self.α, self.i_a, self.m

        v = interp1d(self.grid, v_array, bounds_error=False, 
                     fill_value="extrapolate")
        #𝑘_(𝑡+1)=𝑓(𝛼,𝑘_𝑡)+(𝜃_𝑡+𝑚(1−𝜃_𝑡))(1−𝛿) 𝑘_𝑡 −𝑐_𝑡−𝑖_(𝑎,𝑡)
        return u(c,σ) + β * v(f(y,α) + (θ + m * (1-θ)) * (1 - 𝛿) * y - c - i_a)



def update_bellman(v, bell):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example
    https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    
    The Bellman operator.  Updates the guess of the value function
    and also computes a v-greedy policy.

      * bell is an instance of Bellman equation
      * v is an array representing a guess of the value function

    """
    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)
    
    for i in range(len(bell.grid)):
        y = bell.grid[i]
        # Maximize RHS of Bellman equation at state y
        
        c_star, v_max = maximize(bell.value, min([1e-8,y*0.00001]), 
                                 income_function(y,bell.α)+bell.θ*(1-𝛿)*y-bell.i_a, (y, v))

        v_new[i] = v_max
        v_greedy[i] = c_star

    return v_greedy, v_new

def which_bellman(agentinfo):
    """
    Solves bellman for each affordable adaptation option.
    """
    feasible=[]


    for option in agentinfo.adapt:
        if option[1]>=(income_function(agentinfo.k,agentinfo.α)+agentinfo.θ*(1-𝛿)*agentinfo.k)*0.99990:
            # ensures that the gridpoint
            # just below income, income*0.99999, is included
            pass
        else:
            #print(f'working theta = {agentinfo.θ + option[0] *\
            #  (1-agentinfo.θ)}, i_a= {option[1]}, k= {agentinfo.k}')
            c,v,convergence=solve_bellman(BellmanEquation(u=utility, 
                              f=income_function, k=agentinfo.k, 
                              θ=agentinfo.θ, σ=agentinfo.σ, 
                              α=agentinfo.α, i_a=option[1],m=option[0]))
            feasible.append([v,c,option[1],option[0],convergence])
    
    return feasible



def solve_bellman(bell,
                  tol=0.001,
                  min_iter=10,
                  max_iter=3000,
                  verbose=False):
    """
    From: https://python.quantecon.org/optgrowth.html (similar example
    https://macroeconomics.github.io/Dynamic%20Programming.html#.ZC13-exBy3I)
    
    Solve model by iterating with the Bellman operator.

    """


    # Set up loop
    convergence=True
    v = bell.u(bell.grid,bell.σ)  # Initial condition
    i = 0
    error = tol + 1
    max_error = tol*10 + 1
    while (i < max_iter and (error > tol or max_error > tol*10)) or (i < min_iter):
        v_greedy, v_new = update_bellman(v, bell)
        error = np.abs(v[bell.index] - v_new)[bell.index]
        max_error = np.max(np.abs(v - v_new))
        i += 1
        # if verbose and i % print_skip == 0:
        #     print(f"Error at iteration {i} is {error}.")
        v = v_new

    if (error > tol) or (max_error > tol*10):
        convergence=False
        print(f"{bell.name} failed to converge for k={bell.k}, α = {bell.α},σ ={bell.σ}, i_a={bell.i_a}, and modified θ = {bell.θ + bell.m * (1-bell.θ)} after {i} iterations!")
    elif verbose:
        print(f"Converged in {i} iterations.")
        print(f"Effective k and new c {np.around(bell.grid[bell.index],3),v_greedy[bell.index]}.")
        

    return v_greedy[bell.index],v[bell.index],convergence

class agent:
    def __init__(self,
                    k,            # current state k_t
                    θ,            # perceived shock factor θ
                    σ,            # risk averseness
                    α,            # human capital
                    adapt):       # AdapTable 
        self.k,self.θ, self.σ, self.α, self.adapt=k,θ,σ,α,adapt
        
def iterative_function(data):
    ResultsN=pd.DataFrame({'Agent':[],'Value':[],'Consumption':[], 'i_a':[], 'm':[], 'Converged':[]})
    ResultsL=pd.DataFrame({'Agent':[],'Value':[],'Consumption':[], 'i_a':[], 'm':[],'Converged':[]})
    ResultsH=pd.DataFrame({'Agent':[],'Value':[],'Consumption':[], 'i_a':[], 'm':[],'Converged':[]})
    for i in range(len(data["k"])):
        info=agent(k=data.loc[i,"k"], θ=data.loc[i,"Theta"], σ=data.loc[i,"Sigma"], α=data.loc[i,"Alpha"],adapt=AdapTableArray)
        print(f'{i+1}/{len(data)},k={data.loc[i,"k"]}, θ={data.loc[i,"Theta"]}, σ={data.loc[i,"Sigma"]}, α={data.loc[i,"Alpha"]}')
        feasible=which_bellman(info)
        if len(feasible)>2:
            if np.isnan(feasible[2][0]):
                feasible[2][4] = "OptError"
            ResultsH.loc[i]=[f"S{i}",*feasible[2]]
        if len(feasible)>1:
            if np.isnan(feasible[1][0]):
                feasible[1][4] = "OptError"
            ResultsL.loc[i]=[f"S{i}",*feasible[1]]
        if np.isnan(feasible[0][0]):
                feasible[0][4] = "OptError"
        ResultsN.loc[i]=[f"S{i}",*feasible[0]]

    results=pd.DataFrame(columns=["i_a","consumption"])

    max_index = pd.DataFrame({"N":ResultsN["Value"], "L":ResultsL["Value"], "H":ResultsH["Value"]}).apply('idxmax', axis=1)
    results["i_a"] = max_index
    consumption = pd.DataFrame({"N":ResultsN["Consumption"], "L":ResultsL["Consumption"], "H":ResultsH["Consumption"]})
    results["consumption"] = [consumption.loc[i, max_index.loc[i]] for i in max_index.index]
    return results
    

def nn_function(data): 
    i_a_dict = {"N":0.0,"L":0.2,"H":0.5}
    estimator,cons_scale, i_a_scale,input_scale = load_consumption_model(nn_path,device)  

    estimator.to(device)
    estimator.eval()
    input = data
    print(input)
    # Scale inputs as specified in nn config
    if {"alpha","Alpha"} & input_scale.keys():
        input[:,0]=scale_input(input[:,0], input_scale.get(list({"alpha","Alpha"} & input_scale.keys())[0]),"alpha",False)
    if {"k","K"} & input_scale.keys():
        input[:,1]=scale_input(input[:,1], input_scale.get(list({"k","K"} & input_scale.keys())[0]),"k",False)
    if {"sigma","Sigma"} & input_scale.keys():
        input[:,2]=scale_input(input[:,2], input_scale.get(list({"sigma","Sigma"} & input_scale.keys())[0]),"sigma",False)
    if {"theta","Theta"} & input_scale.keys():
        input[:,3]=scale_input(input[:,3], input_scale.get(list({"theta","Theta"} & input_scale.keys())[0]),"theta",False)
    # Forward pass to get predictions
    with torch.no_grad():

        pred=estimator(input)
    
    results=pd.DataFrame(columns=["i_a","consumption"])
    idx_matches =torch.argmin(torch.abs(pred[:, 0].unsqueeze(1)*i_a_scale - torch.tensor(list(i_a_dict.values()),dtype=torch.float32).unsqueeze(0).to(device)), dim=1).cpu().numpy()
    results["i_a"]=[list(i_a_dict.keys())[idx] for idx in idx_matches]
    #print(f"Setting {sum(results['consumption']<0)} negative consumption predictions to zero,{sum(results['consumption']<-0.1)} were less than -0.1 .")
    results["consumption"]=(pred[:,1]*cons_scale).clamp_(min=0).cpu().numpy()
    return results
    


# Compare the performance of function_a and function_b and save results to CSV
compare_functions(iterative_function, nn_function, input_sizes, seed_quantity, output_file)