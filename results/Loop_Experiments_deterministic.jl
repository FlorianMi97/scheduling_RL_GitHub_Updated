using StatsPlots

using Flux
using ParameterSchedulers
using SchedulingRL

instance = "lhmhlh-4_ii_Dete5"

actionspace = "AIA"
# set up problem
Env = createenv(as = actionspace, instanceName = instance, deterministic = true)

# setup test instances
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs = generatetestset(testenvs, 1, deterministic = true, actionspace = actionspace)

sequence_AIAR = createagent(createExpSequence(instance[1:end-6] ,Env, actionspace), actionspace)
result_seq_AIAR = testagent(sequence_AIAR, testenvs)

println("The average PI gap is: ", result_seq_AIAR[6])

actionspace = "AIM"
Env = createenv(as = actionspace, instanceName = instance, deterministic = true)

# setup test instances
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs = generatetestset(testenvs, 1, deterministic = true, actionspace = actionspace)

AgentGP = createagent(createGP([Env], stationary_samples = true, samples_generation = 1, markerdensity = true), actionspace)
resultsGP = trainagent!(AgentGP,generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs, TBlog = true)


error("stop here")

#--------------------------------------------------------------------------------------------------------------
# WPF
#--------------------------------------------------------------------------------------------------------------
# prio = Priorityfunction("PT + RO + RW + ON + EW + JWM + ST + NI + NW + SLA + CW + CJW + TT")
# # prio = Priorityfunction("RW + ON + ST + SLA + TT")
# # prio = Priorityfunction("PT + ON") to test if arbitrary functions work

# output = numberweights(prio)
# AgentWPF = createagent(
#     createWPF( 
#             ActorCritic(
#                 actor = GaussianNetwork(
#                     pre = Chain(Dense(statesize(Env) => 300, relu),
#                                         Dense(300 => 300, relu),
#                                         Dense(300 => 150, relu),
#                                         Dense(150 => 60, relu),
#                                         Dense(60 => 30)),
#                     μ = Chain(Dense(30, output, tanh)),
#                     σ = Chain(Dense(30, output, sigmoid)),
#                     ),
#                 critic = Chain(Dense(statesize(Env) => 300, relu),
#                                 Dense(300 => 300, relu),
#                                 Dense(300 => 150, relu),
#                                 Dense(150 => 60, relu),
#                                 Dense(60 => 30), Dense(30 => 1)),
#                 optimizer = Adam(5f-6)),
#                 [Env],
#                 [],
#                 prio,
#                 masked = false,
#                 n_steps=2000,
#                 entropy_loss_weight = 0,
#                 #lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2)),
#                 critic_loss_weight = 5f-2,
#                 only_done = true
#             )
#         , "AIM")

# println("start WPF training")
# resultsWPF = trainagent!(AgentWPF,generations = 20, evalevery = 100, finaleval = true, testenvs = testenvs, TBlog = true) #, showprogressbar = false) #, TBlog = true)
# # println("The average PI gap is: ", resultsWPF[6])
# # println("The worst gap is: ", resultsWPF[8])
# # println("All gaps are: \n", resultsWPF[10])
#--------------------------------------------------------------------------------------------------------------
# E2E
#--------------------------------------------------------------------------------------------------------------

# AgentE2E = createagent(
#     createE2E( 
#             ActorCritic(
#                 actor = Chain(Dense(statesize(Env) => 300, relu),
#                         Dense(300 => 300, relu), Dense(300, 300, relu), Dense(300 => actionsize(Env))), # no specified activation function = linear activation
#                 critic = Chain(Dense(statesize(Env) => 300, relu),
#                         Dense(300 => 300, relu),Dense(300 => 150, relu), Dense(150, 60, relu), Dense(60,1)),
#                 optimizer = Adam(1f-5)),            # no specified activation function = linear activation
#                 [Env],
#                 [],
#                 #lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2)),
#             critic_loss_weight = 5f-2,
#             n_steps=2000,
#             entropy_loss_weight = 0,
#             only_done = true
#             )
#         , "AIM")

# println("start E2E training")
# resultsE2E = trainagent!(AgentE2E,generations = 2000, evalevery = 100, finaleval = true, testenvs = testenvs, TBlog = true) #, TBlog = true)    
# println("The average PI gap is: ", resultsE2E[6])
# println("The worst gap is: ", resultsE2E[8])
# println("All gaps are: \n", resultsE2E[10])

#--------------------------------------------------------------------------------------------------------------
# GP
#--------------------------------------------------------------------------------------------------------------

# train GP
# println("start GP training")
# AgentGP = createagent(createGP([Env], stationary_samples = true, samples_generation = 1, markerdensity = true), "AIM")
# resultsGP = trainagent!(AgentGP,generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs, TBlog = true)
    
# println("The average PI gap is: ", resultsGP[6])
# println("The worst gap is: ", resultsGP[8])
# println("All gaps are: \n", resultsGP[10])

# benchmark
# BenchmarkSPT = createagent(createRule(Priorityfunction("TT")),"AIM")
# resultrule = testagent(BenchmarkSPT,testenvs)
# println("The average PI gap is: ", resultrule[6])

# # boxplot with gaps of all 4 models
# boxplot([resultsGP[10],resultsE2E[10],resultsWPF[10],resultrule[10]], 
#     label = ["GP" "E2E" "WPF" "SPT"],
#     title = "Boxplot of gaps",
#     ylabel = "gap",
#     xlabel = "model")

# boxplot([resultsGP[10],resultsE2E[10],resultrule[10]], 
#             label = ["GP" "E2E" "SPT"],
#             title = "Boxplot of gaps",
#             ylabel = "gap",
#             xlabel = "model")

# set up problem
Env = createenv(as = "AIA", instanceName = instance,  deterministic = true)
# setup test instances
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs = generatetestset(testenvs, 1, actionspace = "AIA", deterministic = true)

# prio = Priorityfunction("PT + RO + RW + ON + EW + JWM + ST + NI + NW + SLA + CW + CJW + TT")
# # prio = Priorityfunction("RW + ON + ST + SLA + TT")
# # prio = Priorityfunction("PT + ON") to test if arbitrary functions work


# output = numberweights(prio)
# AgentWPF = createagent(
#     createWPF( 
#             ActorCritic(
#                 actor = GaussianNetwork(
#                     pre = Chain(Dense(statesize(Env) => 300, relu),
#                                         Dense(300 => 300, relu),
#                                         Dense(300 => 150, relu),
#                                         Dense(150 => 60, relu),
#                                         Dense(60 => 30)),
#                     μ = Chain(Dense(30, output, tanh)),
#                     σ = Chain(Dense(30, output, sigmoid)),
#                     ),
#                 critic = Chain(Dense(statesize(Env) => 300, relu),
#                                 Dense(300 => 300, relu),
#                                 Dense(300 => 150, relu),
#                                 Dense(150 => 60, relu),
#                                 Dense(60 => 30), Dense(30 => 1)),
#                 optimizer = Adam(5f-6)),
#                 [Env],
#                 [],
#                 prio,
#                 masked = false,
#                 n_steps=2000,
#                 entropy_loss_weight = 0,
#                 #lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2)),
#                 critic_loss_weight = 5f-2,
#                 only_done = true
#             )
#         , "AIA")

# println("start WPF training")
# resultsWPF = trainagent!(AgentWPF,generations = 2000, evalevery = 100, finaleval = true, testenvs = testenvs, TBlog = true)
# # println("The average PI gap is: ", resultsWPF[6])
# # println("The worst gap is: ", resultsWPF[8])
# # println("All gaps are: \n", resultsWPF[10])

#--------------------------------------------------------------------------------------------------------------
# E2E
#--------------------------------------------------------------------------------------------------------------

# AgentE2E = createagent(
#     createE2E( 
#             ActorCritic(
#                 actor = Chain(Dense(statesize(Env) => 300, relu),
#                         Dense(300 => 300, relu), Dense(300, 300, relu), Dense(300 => actionsize(Env))), # no specified activation function = linear activation
#                 critic = Chain(Dense(statesize(Env) => 300, relu),
#                         Dense(300 => 300, relu),Dense(300 => 150, relu), Dense(150, 60, relu), Dense(60,1)),
#                 optimizer = Adam(1f-5)),            # no specified activation function = linear activation
#                 [Env],
#                 [],
#                 #lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2)),
#             critic_loss_weight = 5f-2,
#             n_steps=2000,
#             entropy_loss_weight = 0,
#             only_done = true
#             )
#         , "AIA")

# println("start E2E training")
# resultsE2E = trainagent!(AgentE2E,generations = 5000, evalevery = 250, finaleval = true, testenvs = testenvs, TBlog = true) #, TBlog = true) 
    
# println("The average PI gap is: ", resultsE2E[6])
# println("The worst gap is: ", resultsE2E[8])
# println("All gaps are: \n", resultsE2E[10])

#--------------------------------------------------------------------------------------------------------------
# GP
#--------------------------------------------------------------------------------------------------------------

# train GP
println("start GP training")
AgentGP = createagent(createGP([Env], stationary_samples = true, samples_generation = 1, markerdensity = true), "AIA")
resultsGP = trainagent!(AgentGP,generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs, TBlog = true)
    
# println("The average PI gap is: ", resultsGP[6])
# println("The worst gap is: ", resultsGP[8])
# println("All gaps are: \n", resultsGP[10])

# benchmark
BenchmarkSPT = createagent(createRule(Priorityfunction("TT")),"AIA")
resultrule = testagent(BenchmarkSPT,testenvs)
# println("The average PI gap is: ", resultrule[6])
# println("The worst gap is: ", resultrule[8])
# println("All gaps are: \n", resultrule[10])

# boxplot with gaps of all 4 models
# boxplot([resultsGP[10],resultsE2E[10],resultsWPF[10],resultrule[10]], 
#     label = ["GP" "E2E" "WPF" "SPT"],
#     title = "Boxplot of gaps",
#     ylabel = "gap",
#     xlabel = "model")

boxplot([resultsGP[10],resultrule[10]], 
            label = ["GP" "SPT"],
            title = "Boxplot of gaps",
            ylabel = "gap",
            xlabel = "model")