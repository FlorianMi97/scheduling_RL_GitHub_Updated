using StatsPlots
using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# Features implemented
# JOB FEATURES
        # "DD"  :  job due date -> never updated use ones.
        # "RO"  :  remaining operations of job 
        # "RW"  :  remaining work of job
        # "ON"  :  average operating time of next operation of job
        # "JS"  :  job slack time (DD - RW - CT)
        # "RF"  :  routing flexibility of remaining operations of job

# MACHINES FEATURES
        # "EW"  :  expected future workload of a machine (proportionally if multiple possible machines)
        # "CW"  :  current workload of a machine (proportionally if multiple possible machines)
        # "JWM" :  Future jobs waiting for machine (proportionally if multiple possible machines)
        # "CJW" :  current jobs waiting (proportionally if multiple possible machines)

# JOB-MACHINES FEATURES
        # "TT"  :  total time of job machine pair (including waiting idle setup processing)
        # "PT"  :  processing time of job machine pair
        # "ST"  :  setup time of job machine pair
        # "NI"  :  required idle time of a machine
        # "NW"  :  needed waiting time of a job
        # "SLA" :  binary: 1 if setupless alternative is available when setup needs to be done, 0 otherwise
        # "BSA" :  binary: 1 if better setup alternative is available, 0 otherwise
        # "DBA" :  returns possitive difference between setup time of current setup and best alternative

# GENERAL
        # "CT"  :  current time

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "useCase_2_stages"
Env_AIM = createenv(as = "AIM",instanceType = "usecase", instanceName = instance)
# Env_AIA = createenv(as = "AIA",instanceType = "usecase", instanceName = instance)
# Env_AIAR = createenv(as = "AIAR",instanceType = "usecase", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "usecase" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")
# testenvs_AIA = generatetestset(testenvs, 100, actionspace = "AIA")
# testenvs_AIAR = generatetestset(testenvs, 100, actionspace = "AIAR")

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------



# Number of Tardy Jobs

# SIMPLE RULE
println("\n", "DD + RO + EW + PT + ST", "\n")
SPT_AIM = createagent(createRule(Priorityfunction("DD + RO + EW + PT + ST")),"AIM")
result_rule_AIM = testagent(SPT_AIM, testenvs_AIM)
println("Result of Rule without specifying weights: ", result_rule_AIM[1], "\n")

# # GES
priofunction = Priorityfunction("DD + RO + EW + PT + ST")
ges= createGES("global",[Env_AIM], priofunction)

Agent_GES_AIM = createagent(ges, "AIM", obj = [1.0,0.0,0.0,0.0,0.0,0.0])

println("GES weights before: ", Agent_GES_AIM.approximator.bestweights, "\n")

for env in testenvs_AIM
        result_GES_for_check = testGES(Agent_GES_AIM, ges.priorityfunctions, ges.bestweights, env, Agent_GES_AIM.objective, 100, Agent_GES_AIM.rng)
        println("Result of Rule after initializing weights: ", result_GES_for_check[1], "\n")
end

results_GES_AIM = trainagent!(Agent_GES_AIM, generations = 10, evalevery = 1, finaleval = true, showinfo = true, testenvs = testenvs_AIM)
#println("GES weights after: ", Agent_GES_AIM.approximator.bestweights)
println("GES Result: ", results_GES_AIM[1])

# # boxplot results
#graph_Flow_time= boxplot([result_rule_AIM[10], results_GES_AIM[10]],
#         label = ["DD + PT" "GES"],
#         title = "Gaps to optimal makespan",
#         ylabel = "gap",
#bv          xlabel = "model")



#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

# Total Tardiness


#--------------------------------------------------------------------------------------------------------------
# SIMPLE RULE: EDD
#EDD_AIM = createagent(createRule(Priorityfunction("DD")),"AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
#result_rule_AIM = testagent(EDD_AIM,testenvs_AIM)
#println("EDD AIM: ", result_rule_AIM[6])

#--------------------------------------------------------------------------------------------------------------
#GP

#=

GP = createGP([Env_AIM],simplerules = true)
AgentGP = createagent(GP, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])

resultsGP = trainagent!(AgentGP,generations = 1, evalevery = 0, finaleval = true, testenvs = testenvs_AIM)
println("GP AIM Result: ", resultsGP[1])




#--------------------------------------------------------------------------------------------------------------

# # GES

#------------------------------------------
#granularity = "global"

#AgentGP.model[1][1][1]
#priorules = Priorityfunction("TT + EW + RW + JS + DD + PT + ST")

#------------------------------------------
#granularity = "stage"
#AgentGP.model[1][1]
#priorule1 = Priorityfunction("TT + EW + RW + JS + DD")
#priorule2 = Priorityfunction("TT + EW + RW + JS + DD + ST")
#priorules = Vector([priorule1, priorule2])
#------------------------------------------
# granulariry = "resource"

#------------------------------------------
#Get GP output to initialize GES with a simplified (linear) version of it


GP_best_priority_function = get_expression(AgentGP.model[1][1][1],AgentGP.approximator)
println("GP best priority function: ", AgentGP.model[1][1][1], "\n")

Simpliefied_best_Rule = simplify_expression(GP_best_priority_function,AgentGP.approximator)
println("Simpliefied_best_Rule: \n", Simpliefied_best_Rule , "\n")
drawTree(Simpliefied_best_Rule,1)


#Create a priorityfunction from the simplified best rule (priorules::expr, priofunction_strig::String)
println("Parameters for camparison: ", "\n")
priorules, priofunction_string = PriorityFunctionFromTree(Simpliefied_best_Rule, true) 


# create GES
ges = createGES("global", [Env_AIM], priorules)

#------------------------------------------
# initial parameters and setup GES

Analyze_simplified_Tree = analyze_tree(Simpliefied_best_Rule)
Initial_best_weights = Covert_weights_from_GP(Analyze_simplified_Tree[3], priofunction_string)
s = Vector{Vector{Float32}}([fill(log(3), numberweights(priorules))]) # source-paper sets all sigma to log(3)
#optimizer = "ADAM"

setinitalparameters!(ges, initialweights = Initial_best_weights, sigma = s, )
println("Mask of priorityfunction: ", priorules.nn_mask, "\n")
println("Constant of priorityfunction: ", priorules.constant, "\n")
println("GES initial best weights: ", ges.bestweights, "\n")
println("GES initial sigma: ", ges.sigma, "\n")


#------------------------------------------
# create agent and learning process 

#benchmark initial rule
Agent_Benchmark_Rule = createagent(createRule(priorules),"AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
result_EDD_AIM = testagent(Agent_Benchmark_Rule, testenvs_AIM)
println("Agent_Benchmark_Rule Result: ", result_EDD_AIM[6])


Agent_GES_AIM = createagent(ges, "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
results_GES_AIM = trainagent!(Agent_GES_AIM, generations = 10, evalevery = 0, finaleval = true, testenvs = testenvs_AIM)
println("weights after: ", ges.bestweights[1])
println("GES AIM Result: " , results_GES_AIM[1])
#boxplot results
#=
boxplot([result_rule_AIM[10], results_GES_AIM[10]],
        label = ["Benchmark" "GES"],
        title = "Differences to optimal total tardiness",
        ylabel = "difference",
        xlabel = "model")
=#
boxplot([result_EDD_AIM[10],resultsGP[10], results_GES_AIM[10]],
        label = ["Benchmark" "GP_input" "GES_after"],
        title = "Differences to optimal total tardiness",
        ylabel = "difference",
        xlabel = "model")







=#







#--------------------------------------------------------------------------------------------------------------
# TEST AND TRY

# Test if the best priority function is evaluated the same way as the simplified one --> yes
#=
GP_best_priority_function = get_expression([SchedulingRL.add, SchedulingRL.add, SchedulingRL.add, :CT, nothing, nothing, nothing, nothing, nothing, nothing, :ST, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.add, :JS, nothing, nothing, nothing, nothing, nothing, nothing, :RF, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.add, SchedulingRL.add, 0.02, nothing, nothing, nothing, nothing, nothing, nothing, :JS, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.sub, 0.413, nothing, nothing, nothing, nothing, nothing, nothing, :RF, nothing, nothing, nothing, nothing, nothing, nothing],AgentGP.approximator)
println("GP best priority function: ", GP_best_priority_function, "\n")
test_of_best_rule = testfitness([GP_best_priority_function], testenvs_AIM[1], AgentGP.objective, 30, AgentGP.rng)[1:3]
println("Test Result Priority Rule: ", test_of_simplified_rule[1], "\n")

Simpliefied_best_Rule = simplify_expression(GP_best_priority_function,AgentGP.approximator)
println("Simpliefied_best_Rule: \n", Simpliefied_best_Rule , "\n")
#drawTree(Simpliefied_best_Rule,1)
test_of_simplified_rule = testfitness([Simpliefied_best_Rule], testenvs_AIM[1], AgentGP.objective, 30, AgentGP.rng)[1:3]
println("Test Result of simplified rule: ", test_of_simplified_rule[1], "\n")
=#
#------------------------------------------

# Testing of differnt stages and the recieved actionvector in the GP

#=
GP = createGP([Env_AIM],simplerules = true)
AgentGP = createagent(GP, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])

example_tree_array_1 = Any[SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, :CT, :CW, SchedulingRL.sub, 0.378, :PT, SchedulingRL.sub, SchedulingRL.add, :RW, 0.131, SchedulingRL.add, :NI, :PT, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, :RO, :CJW, SchedulingRL.add, 0.462, :CT, SchedulingRL.add, SchedulingRL.sub, 0.102, :RW, SchedulingRL.sub, 0.884, 0.112, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, :RF, :BSA, SchedulingRL.sub, :ON, 0.699, SchedulingRL.sub, SchedulingRL.add, :DBA, :RO, SchedulingRL.sub, :CJW, :ST, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.sub, :CW, :TT, SchedulingRL.add, :PT, :CW, SchedulingRL.sub, SchedulingRL.sub, :SLA, 0.592, SchedulingRL.add, :RO, :CT]
example_tree_array_2 = Any[SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, :JWM, :TT, SchedulingRL.add, :RF, :CW, SchedulingRL.sub, SchedulingRL.sub, :JWM, 0.593, SchedulingRL.add, :SLA, :RW, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.sub, 0.255, :RW, SchedulingRL.add, :NI, :NI, SchedulingRL.sub, SchedulingRL.sub, :RO, :RF, SchedulingRL.sub, :CT, :DBA, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, :JWM, :RW, SchedulingRL.add, :NW, :BSA, SchedulingRL.sub, SchedulingRL.add, :ST, 0.907, SchedulingRL.sub, :JWM, :ST, SchedulingRL.add, SchedulingRL.add, SchedulingRL.add, 0.636, :DD, SchedulingRL.sub, :DBA, :DD, SchedulingRL.sub, SchedulingRL.sub, :ST, 0.42, SchedulingRL.sub, 0.272, 0.04]
example_tree_array_3 = Any[SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, :RF, :CJW, SchedulingRL.add, :ST, :ON, SchedulingRL.add, SchedulingRL.add, 0.919, :RF, SchedulingRL.sub, :ST, :EW, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, :BSA, :NI, SchedulingRL.add, :RF, :JS, SchedulingRL.sub, SchedulingRL.add, 0.744, 0.905, SchedulingRL.sub, :RF, 0.74, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, :EW, 0.565, SchedulingRL.add, :SLA, :JWM, SchedulingRL.sub, SchedulingRL.add, 0.691, :BSA, SchedulingRL.add, :PT, :JS, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, :BSA, :JS, SchedulingRL.add, :TT, :BSA, SchedulingRL.sub, SchedulingRL.add, 0.354, 0.251, SchedulingRL.sub, 0.188, :PT]
example_tree_array_4 = Any[SchedulingRL.add, SchedulingRL.sub, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, 0.821, 0.318, SchedulingRL.add, :RW, :JWM, SchedulingRL.sub, SchedulingRL.add, :SLA, 0.613, SchedulingRL.add, :DD, :ON, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.sub, :EW, :EW, SchedulingRL.add, :DD, :DD, SchedulingRL.add, SchedulingRL.add, :DD, 0.879, SchedulingRL.add, :TT, :ST, SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, SchedulingRL.add, :NW, :PT, SchedulingRL.add, :CJW, :NI, SchedulingRL.add, SchedulingRL.add, :PT, :RF, SchedulingRL.sub, :ST, :ST, SchedulingRL.add, SchedulingRL.add, SchedulingRL.sub, :DD, :ON, SchedulingRL.sub, :RF, :NI, SchedulingRL.add, SchedulingRL.sub, :DD, :RW, SchedulingRL.sub, :ST, :TT]

#individuum = [example_tree_array_1]
individuum = [example_tree_array_1, example_tree_array_2, example_tree_array_3, example_tree_array_4]
expression_individdum = [get_expression(individuum[i], GP) for i in eachindex(individuum)]

actionvector = actionsfromindividuum(expression_individdum,Env_AIM)
println("Actionvector: ", actionvector, "\n")

=#

#------------------------------------------
#=

GP = createGP([Env_AIM],simplerules = true)
AgentGP = createagent(GP, "AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])

example_tree_array = build_tree(GP, true)

expr= get_expression(example_tree_array,GP)
#drawTree(expr,1)

simplifiedExpr = simplify_expression(expr,GP)
drawTree(simplifiedExpr,1)

#println("Input Expression: \n", simplifiedExpr , "\n")
#featureToPrimitives, numvalue, FeatureCounter = analyze_tree(simplifiedExpr)
#println("Feature to Primitives: \n", featureToPrimitives, "\n")
#println("Numvalue: ", numvalue, "\n")
#println("FeatureCounter: ", FeatureCounter, "\n")
#drawTree(simplifiedExpr,1)

Prio, Priostring = PriorityFunctionFromTree(simplifiedExpr,5,-5,true)
println("Priofunction Expression: ", Priostring, "\n")
println("Mask: ", Prio.nn_mask, "\n")
println("Constant: ", Prio.constant, "\n")


#priorules = Priorityfunction("TT + EW + RW + JS + DD + PT - ST")
#println("Priofunction Expression: ", priorules.expression, "\n")



#tree_array= Any[SchedulingRL.add, SchedulingRL.sub, SchedulingRL.add, :TT, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.add, SchedulingRL.sub, 0.258, 0.325, SchedulingRL.add, :ON, :NI, SchedulingRL.add, 0.806, nothing, nothing, nothing, nothing, nothing, nothing, SchedulingRL.sub, :CW, nothing, nothing, :NI, nothing, nothing, SchedulingRL.sub, :RF, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, :JS, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing]
#expression= get_expression(tree_array,GP)
#println(expression)
#drawTree(expression,1)
#simplifiedExpr = simplify_expression(expression,GP)
#no_constants = reduceTreeNumericals(simplifiedExpr)
#println(no_constants)
#drawTree(no_constants,1)


#final_population= AgentGP.model[1][1]
#drawTree(get_expression(final_population[1],AgentGP.approximator))
#println(get_expression(final_population[1],AgentGP.approximator))
#println(PriorityfunctionfromTree(final_population[1],AgentGP.approximator)

#------------------------------------------

=#








