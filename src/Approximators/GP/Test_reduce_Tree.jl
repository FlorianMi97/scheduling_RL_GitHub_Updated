using Pkg
"""activate the environment via the toml.file"""
Base.activate_project()
Pkg.activate("GP")

using StatsBase
using Random
using JSON
using Plots
using GraphRecipes
using Distributions
using SparseArrays
using VegaLite
using DataFrames
#using PlotlyJS

include("Utils.jl")
include("Problem.jl")
include("GP.jl")
#include("Simulation.jl")
include("Simulation_changed_updating.jl")
include("Testing.jl")

start = time()
neg(a) = a*(-1); 
squ(a) = a^2; 
"""protected division"""
div(a,b,undef=10e6) = ifelse(b==0, a+undef, /(a,b)); 
inv(a) = div(1,a); 
mul(a,b) = a*b; 
add(a,b) = a+b; 
sub(a,b) = a-b; 

arity = Dict(:add => 2, :sub => 2, :mul => 2, :div => 2, :min => 2, :max => 2, :inv => 1, :neg => 1, :squ => 1)

#define primitives HERE
PRIMITIVES = [add,sub,mul,div,min,max,neg,inv,squ]

primitives1 = [p for p in PRIMITIVES if arity[Symbol(p)] == 1]
primitives2 = [p for p in PRIMITIVES if arity[Symbol(p)] == 2]

primitivesSymbol = Dict(f => string(f) for f in PRIMITIVES)

# Features:
# "PT"  :  processing time of job machine pair
# "DD"  :  job due date
# "RO"  :  remaining operations of job 
# "RW"  :  remaining work of job
# "ON"  :  average operating time of next operation of job
# "JS"  :  job slack time 
# "RF"  :  routing flexibility of remaining operations of job
# "CT"  :  current time
# "EW"  :  expected future workload of a machine (proportionally if multiple possible machines)
# "CW"  :  current workload of a machine (proportionally if multiple possible machines)
# "JWM" :  Future jobs waiting for machine (proportionally if multiple possible machines)
# "CJW" :  current jobs waiting (proportionally if multiple possible machines)
# "ST"  :  setup time of job machine pair
# "RI"  :  required idle time of a machine
# "NW"  :  needed waiting time of a job
# "SLA" :  binary: 1 if setupless alternative is available when setup needs to be done, 0 otherwise

stringFeatures = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "RI", "NW", "SLA", "EW", "JWM","CW", "CJW"] # no "CT"

FEATURES = [Symbol(f) for f in stringFeatures]
FEAT_STRING_DICT = Dict(FEATURES[i] => stringFeatures[i] for i in eachindex(stringFeatures))
FUN_STRING_DICT = Dict(add=>"+", sub=>"-", mul=>"*", div=>"/", min=>"min", max=>"max", neg=>"neg", inv=>"inv", squ=>"squ")
STRING_FUN_DICT = Dict(v=>k for (k,v) in FUN_STRING_DICT)
MAXDEPTH = 5

# obj weights = [makespan, totalFlowTime, tardiness, numberTardyJobs, numSetups]
OBJ_WEIGHTS = [1, 0, 10, 0, 0]

#featureValues = [1,2,3,4,5,6,7,8,9,10,11,12]

function objective(objVector, weightVector)
    return objVector' * weightVector
    # syms=symbols(ex)
    # featDict = Dict(FEATURES[i]=>i for i in 1:length(FEATURES) if FEATURES[i] in syms)
    # return abs(evaluate(ex, featDict, featureValues))
end    

nSamples = 30
instanceName = "useCase_2_stages"
problemFile = string(@__DIR__,"/Instances/",instanceName,"_randomOrderBook.json")
PROBLEMS = Dict(1=>Problem(problemFile, 1, randomOrderbooks = true, mOrders = 55))
INIT = Dict(PROBLEMS[1].id => InitialState(PROBLEMS[1]))
SAMPLES = Dict(1 => sample_uncertainties(PROBLEMS[1], nSamples)) 

#choose simulation method from "VB", "VF", and "VFS"
SIM_METHOD = "VFS"

NSGA_2 = "n"
GENPRO = GP_glo(PRIMITIVES, FEATURES, MAXDEPTH, arity, objective, nSamples, popSize = 150, generations = 100, stages=1, intelRule=true, nsga_2 = NSGA_2)

nGen = 0
timestamp = time()

print(length(build_tree(GENPRO))[1])

# POP = initial_population(GENPRO, PROBLEMS, SAMPLES)
POP = initial_pop_glo()


final = time() - timestamp
println("best value of initial pop is: ", POP[1][2]/nSamples)
println("initial pop took $final")
while nGen != GENPRO.generations
    global nGen += 1 
    # next_gen = next_generation(POP, GENPRO, PROBLEMS, SAMPLES)
    # global POP = new_population(POP, next_gen, GENPRO, PROBLEMS, SAMPLES)
    next_gen = next_generation_glo()
    global POP = new_population_glo(next_gen)
    # global pop = iterated_local_search!(pop, test_gp)
    println("best value after iteration $nGen: ", POP[1][2]/nSamples) 
end

println(POP[1:5])

for nr in 1:5
    println("rule $nr")
    println("original one")
    for i in 1:GENPRO.stages
        println(get_expression(POP[nr][1][i], GENPRO))
    end

    println(featImp(POP[nr][1],GENPRO))
end

