using StatsPlots

using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "useCase_2_stages"
Env_AIA = createenv(as = "AIA",instanceType = "usecase", instanceName = instance)
Env_AIM = createenv(as = "AIM",instanceType = "usecase", instanceName = instance)
Env_AIAR = createenv(as = "AIAR",instanceType = "usecase", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "usecase" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs_AIA = generatetestset(testenvs, 100, actionspace = "AIA")
testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")
testenvs_AIAR = generatetestset(testenvs, 100, actionspace = "AIAR")


# SPT_AIM = createagent(createRule(Priorityfunction("TT")),"AIM")
# result_rule_AIM = testagent(SPT_AIM, testenvs_AIM)
# println("SPT AIM: ", result_rule_AIM)

#--------------------------------------------------------------------------------------------------------------
# Expected sequences
#--------------------------------------------------------------------------------------------------------------
# Exp. sequence AIAR
# sequence_AIAR = createagent(createExpSequence(instance[1:end-6] ,Env_AIAR, "AIAR"), "AIAR",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
# result_seq_AIAR = testagent(sequence_AIAR, testenvs_AIA)
# println("Exp. sequence AIAR: ", result_seq_AIAR[6])
#--------------------------------------------------------------------------------------------------------------
# Priority Rule SPT
#--------------------------------------------------------------------------------------------------------------
# Priority Rule SPT AIM
# EDD_AIM = createagent(createRule(Priorityfunction("DD")),"AIM",obj = [0.0,0.0,1.0,0.0,0.0,0.0])
# result_rule_AIM = testagent(EDD_AIM,testenvs_AIM)
# println("EDD AIM: ", result_rule_AIM[6])

#--------------------------------------------------------------------------------------------------------------
# DRL
#--------------------------------------------------------------------------------------------------------------
# AgentE2E = createagent(
#     createE2E( 
#             ActorCritic(
#                 actor = Chain(Dense(statesize(Env_AIM) => 300, relu),
#                         Dense(300 => 300, relu), Dense(300, 300, relu), Dense(300 => actionsize(Env_AIM))), # no specified activation function = linear activation
#                 critic = Chain(Dense(statesize(Env_AIM) => 300, relu),
#                         Dense(300 => 300, relu),Dense(300 => 150, relu), Dense(150, 60, relu), Dense(60,1)),
#                 optimizer = Adam(1f-5)),            # no specified activation function = linear activation
#                 [Env_AIM],
#                 [],
#                 #lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2)),
#             critic_loss_weight = 5f-2,
#             n_steps=2000,
#             entropy_loss_weight = 0,
#             only_done = true
#             ),
#         "AIM",
#         obj = [0.0,0.0,1.0,0.0,0.0,0.0])

# println("start E2E training")
# resultsE2E = trainagent!(AgentE2E, generations = 5000, evalevery = 250, finaleval = true, testenvs = testenvs_AIM, TBlog = true) #, TBlog = true) 

#--------------------------------------------------------------------------------------------------------------
# GP
#--------------------------------------------------------------------------------------------------------------

# # test cycling of uncertainity
# Agent_GP_AIM_cycling = createagent(createGP([Env_AIM], stationary_samples = false, samples_generation = 12, change_every = 50, steps_in_cycle = 5,
#                                              sample_cycling = true, markerdensity = true), "AIM", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
# results_GP_AIM_cycling = trainagent!(Agent_GP_AIM_cycling, generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs_AIM, TBlog = true)

# test cycling of uncertainity
Agent_GP_AIA_cycling = createagent(createGP([Env_AIA], stationary_samples = false, samples_generation = 12, change_every = 50, steps_in_cycle = 5,
                                             sample_cycling = true, markerdensity = true), "AIA", obj = [0.0,0.0,1.0,0.0,0.0,0.0])
results_GP_AIA_cycling = trainagent!(Agent_GP_AIA_cycling, generations = 500, evalevery = 50, finaleval = true, testenvs = testenvs_AIA, TBlog = true)
#--------------------------------------------------------------------------------------------------------------
# Plots
#--------------------------------------------------------------------------------------------------------------

# boxplot results
boxplot([result_seq_AIAR[10], results_GP_AIA_cycling[10]],
        label = ["Exp. Seq" "EDD" "GP" "DRL"],
        title = "Boxplot of gaps",
        ylabel = "gap",
        xlabel = "model")


