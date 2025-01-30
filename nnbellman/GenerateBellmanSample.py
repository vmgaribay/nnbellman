# Never used
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import sys
sys.path.append("~/.local/lib/python3.8/site-packages")
from SALib.sample import saltelli,latin
import pandas as pd
import numpy as np

from scipy.interpolate import griddata


# Input Variables
# k (capital) 1-30 uniform, 1-15 uniform for random sample 
# θ (perceived shock factor) 0.1-1 uniform
# σ (risk averseness) normal centered around 1.08
# α (aptitude/human capital) normal centered around 1



# %%
# Generate Samples

# Saltelli Sample

n=8192
#N=5000
seed=1
#second order true n(2d+2)=n*10
#second order false n*(d+2)=n*6

problem={   "names": ["Alpha","k","Sigma","Theta"], 
            "num_vars":4,
            "bounds":[[1.08,0.074],[0.01,40],[0.01,2],[0.01,1]],
            "dists":["norm","unif","unif","unif"]}

S_sample=saltelli.sample(problem,n,calc_second_order=False)


S_sampledf=pd.DataFrame(S_sample, columns=["Alpha","k","Sigma","Theta"])

S_sampledf["Sigma"]=np.round(S_sampledf["Sigma"],1)
S_sampledf["Theta"]=np.round(S_sampledf["Theta"],1)
S_sampledf.index.name="AgentID"
# drop duplicates and samples with values of alpha and sigma < 0
S_sampledf=S_sampledf.drop_duplicates()
S_sampledf=S_sampledf[(S_sampledf["Alpha"]>=0) & (S_sampledf["Sigma"]>=0)]

S_sampledf.to_csv("AgentData-Updated30Oct2024.csv")

# %%
# global variables for Bellman equation
𝛿 = 0.08 #depreciation
β = 0.95 #discount factor


TechTable = {#contains values for 0:gamma 1:cost 2:theta
# VMG things get stuck in a while loop if the gamma is less than 0.3 
# (tried 0.2) not sure yet if/how this will be problematic 
# Also important to make sure values are in the correct order
# i.e. that the threshold between medium and high is at a 
# higher k than the threshold between low and medium 
# This can be checked with k_threshold.py

    "low":   [0.3,  0   ],
    "medium":[0.35, 0.15],
    "high":  [0.45, 0.65]}

TechTableArray = np.array([[ 0.3,  0 ],[0.35, 0.15],[0.45, 0.65]])

AdapTable = {
    # contains values for 0:theta 1:cost 
    # (for consideration:effort? type? design life?)
    "none":   [  0, 0   ],
    "good":   [0.75, 0.20],
    "better": [0.95, 0.50]}

AdapTableArray = np.array([[ 0,  0 ],[0.75, 0.2],[0.95, 0.5]])



# Define Optimization Routine:


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
        #VMG HELP! can anyone check that (1) subtracting i_a and 
        # (2) omitting any grid values less than i_a 
        # will not be problematic? The only thing I can come up with
        # is if i_a is greater than k*0.99999
        # which_bellman() now accounts for that case. Whole thing 
        # could use refinement.
        #VMG Dear past self, it became problematic when the resource
        # pool for investment changed to include depreciated capital, 
        # but it is fixed to now accurately reflect the relationship
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

 



# %%
class agent:
    def __init__(self,
                 k,            # current state k_t
                 θ,            # perceived shock factor θ
                 σ,            # risk averseness
                 α,            # human capital
                 adapt):       # AdapTable 
        self.k,self.θ, self.σ, self.α, self.adapt=k,θ,σ,α,adapt

# %%


# Obtain Output for Sample
 
# Saltelli
S1_ResultsN=pd.DataFrame({'Agent':[],'Value':[],'Consumption':[], 'i_a':[], 'm':[], 'Converged':[]})
S1_ResultsL=pd.DataFrame({'Agent':[],'Value':[],'Consumption':[], 'i_a':[], 'm':[],'Converged':[]})
S1_ResultsH=pd.DataFrame({'Agent':[],'Value':[],'Consumption':[], 'i_a':[], 'm':[],'Converged':[]})
for i in S_sampledf.index:#len(S_sampledf)):
    info=agent(k=S_sampledf.loc[i,"k"], θ=S_sampledf.loc[i,"Theta"], σ=S_sampledf.loc[i,"Sigma"], α=S_sampledf.loc[i,"Alpha"],adapt=AdapTableArray)
    print(f'{i}/{max(S_sampledf.index)},k={S_sampledf.loc[i,"k"]}, θ={S_sampledf.loc[i,"Theta"]}, σ={S_sampledf.loc[i,"Sigma"]}, α={S_sampledf.loc[i,"Alpha"]}')
    feasible=which_bellman(info)
    if len(feasible)>2:
        if np.isnan(feasible[2][0]):
            feasible[2][4] = "OptError"
        S1_ResultsH.loc[i]=[f"S{i}",*feasible[2]]
    if len(feasible)>1:
        if np.isnan(feasible[1][0]):
            feasible[1][4] = "OptError"
        S1_ResultsL.loc[i]=[f"S{i}",*feasible[1]]
    if np.isnan(feasible[0][0]):
            feasible[0][4] = "OptError"
    S1_ResultsN.loc[i]=[f"S{i}",*feasible[0]]
    
print(pd.DataFrame({"N":S1_ResultsN["Value"], "L":S1_ResultsL["Value"], "H":S1_ResultsH["Value"]}))
S1_ResultsN.to_csv("ResultsNone-Efficacy-29Jan2025.csv")
S1_ResultsL.to_csv("ResultsLow-Efficacy-29Jan2025.csv")
S1_ResultsH.to_csv("ResultsHigh-Efficacy-29Jan2025.csv")

S1_Index = pd.DataFrame({"N":S1_ResultsN["Value"], "L":S1_ResultsL["Value"], "H":S1_ResultsH["Value"]}).apply('idxmax', axis=1)
S1_Cons = pd.DataFrame({"N":S1_ResultsN["Consumption"], "L":S1_ResultsL["Consumption"], "H":S1_ResultsH["Consumption"]})

S1_Result = [S1_Cons.loc[i, S1_Index.loc[i]] for i in S1_Index.index]
S1_Final= pd.DataFrame({"Equation":S1_Index,"Consumption": S1_Result})

S1_Final.to_csv("ResultsFinal-Efficacy-29Jan2025.csv")
    


'''
S1_ResultsN=pd.read_csv("ResultsNone-Updated12Oct2024.csv",index_col=1)
S1_ResultsL=pd.read_csv("ResultsLow-Updated12Oct2024.csv",index_col=1)
S1_ResultsH=pd.read_csv("ResultsHigh-Updated12Oct2024.csv",index_col=1)
print(pd.DataFrame({"N":S1_ResultsN["Value"], "L":S1_ResultsL["Value"], "H":S1_ResultsH["Value"]}))'''

# %%
#postprocessing to remove points that did not converge or caused an optimization error

N_ErrorIndex=S1_ResultsN["Agent"][S1_ResultsN["Converged"]!=True]
L_ErrorIndex=S1_ResultsL["Agent"][S1_ResultsL["Converged"]!=True]
H_ErrorIndex=S1_ResultsH["Agent"][S1_ResultsH["Converged"]!=True]


ErrorIndex=list(pd.concat([N_ErrorIndex,L_ErrorIndex,H_ErrorIndex]))
S1_ResultsNclean=S1_ResultsN[~S1_ResultsN["Agent"].isin(ErrorIndex)]
S1_ResultsLclean=S1_ResultsL[~S1_ResultsL["Agent"].isin(ErrorIndex)]
S1_ResultsHclean=S1_ResultsH[~S1_ResultsH["Agent"].isin(ErrorIndex)]


S1_Indexclean = pd.DataFrame({"N":S1_ResultsNclean["Value"], "L":S1_ResultsLclean["Value"], "H":S1_ResultsHclean["Value"]}).apply('idxmax', axis=1)
S1_Consclean = pd.DataFrame({"N":S1_ResultsNclean["Consumption"], "L":S1_ResultsLclean["Consumption"], "H":S1_ResultsHclean["Consumption"]})

S1_Resultclean = [S1_Consclean.loc[i, S1_Indexclean.loc[i]] for i in S1_Indexclean.index]
S1_Finalclean= pd.DataFrame({"Equation":S1_Indexclean,"Consumption": S1_Resultclean})
S1_Finalclean.index.name="AgentID"

S1_Finalclean.to_csv("ResultsFinal-Efficacy-29Jan2025clean.csv")
    
print(f' Percent retained:{len(S1_Finalclean)/len(S1_Final)*100}%')




# %%
'''
S1_ResultsN=pd.read_csv("ResultsNone-Updated30Oct2024.csv",index_col=0)
print(S1_ResultsN)
N_ErrorIndex=S1_ResultsN["Agent"][S1_ResultsN["Converged"]!=True]

S1_ResultsNclean=S1_ResultsN[~S1_ResultsN["Agent"].isin(N_ErrorIndex)]

S1_Finalclean= pd.DataFrame({"Equation":"N","Consumption": S1_ResultsNclean["Consumption"]})
S1_Finalclean.index.name="AgentID"
S1_Finalclean.to_csv("ResultsNone-Updated30Oct2024clean.csv")

'''


