using StatsPlots

using Flux
using ParameterSchedulers
using SchedulingRL

#--------------------------------------------------------------------------------------------------------------
# setup environment
#--------------------------------------------------------------------------------------------------------------
instance = "lhmhlh-4_ii_Expo5"
Env_AIA = createenv(as = "AIA", instanceName = instance)
Env_AIM = createenv(as = "AIM", instanceName = instance)
Env_AIAR = createenv(as = "AIAR", instanceName = instance)
#--------------------------------------------------------------------------------------------------------------
# setup test environments with samples
#--------------------------------------------------------------------------------------------------------------
testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>instance)]

testenvs_AIA = generatetestset(testenvs, 100, actionspace = "AIA")
testenvs_AIM = generatetestset(testenvs, 100, actionspace = "AIM")
testenvs_AIAR = generatetestset(testenvs, 100, actionspace = "AIAR")

#--------------------------------------------------------------------------------------------------------------
# DRL
#--------------------------------------------------------------------------------------------------------------
AgentE2E = createagent(
    createE2E( 
            ActorCritic(
                actor = Chain(Dense(statesize(Env_AIM) => 300, relu),
                        Dense(300 => 300, relu), Dense(300, 300, relu), Dense(300 => actionsize(Env_AIM))), # no specified activation function = linear activation
                critic = Chain(Dense(statesize(Env_AIM) => 300, relu),
                        Dense(300 => 300, relu),Dense(300 => 150, relu), Dense(150, 60, relu), Dense(60,1)),
                optimizer = Adam(1f-5)),            # no specified activation function = linear activation
                [Env_AIM],
                [],
                #lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2)),
            critic_loss_weight = 5f-2,
            n_steps=2000,
            entropy_loss_weight = 0,
            only_done = true
            )
        , "AIM")

println("start E2E training")
resultsE2E = trainagent!(AgentE2E,generations = 5000, evalevery = 250, finaleval = true, testenvs = testenvs_AIM, TBlog = true) #, TBlog = true) 

#--------------------------------------------------------------------------------------------------------------
# GP
#--------------------------------------------------------------------------------------------------------------

# test cycling of uncertainity
Agent_GP_AIM_cycling = createagent(createGP([Env_AIM], stationary_samples = false, samples_generation = 12, change_every = 50, steps_in_cycle = 5,
                                             sample_cycling = true, markerdensity = true), "AIM")
results_GP_AIM_cycling = trainagent!(Agent_GP_AIM_cycling, generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs_AIM, TBlog = true)

# # test new samples constantly
# Agent_GP_AIM_samples = createagent(createGP([Env_AIM], stationary_samples = false, samples_generation = 5, change_every = 10,
#                                              markerdensity = true), "AIM")
# results_GP_AIM_samples = trainagent!(Agent_GP_AIM_samples, generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs_AIM, TBlog = true)

# # stationary samples
# Agent_GP_AIM = createagent(createGP([Env_AIM], stationary_samples = true, samples_generation = 30, markerdensity = true), "AIM")
# results_GP_AIM = trainagent!(Agent_GP_AIM, generations = 1000, evalevery = 50, finaleval = true, testenvs = testenvs_AIM, TBlog = true)

#--------------------------------------------------------------------------------------------------------------
# Expected sequences
#--------------------------------------------------------------------------------------------------------------
# Exp. sequence AIA
sequence_AIA = createagent(createExpSequence(instance[1:end-6] ,Env_AIA, "AIA"), "AIA")
result_seq_AIA = testagent(sequence_AIA, testenvs_AIA)
println("Exp. sequence AIA: ", result_seq_AIA[6])

# Exp. sequence AIAR
sequence_AIAR = createagent(createExpSequence(instance[1:end-6] ,Env_AIAR, "AIAR"), "AIAR")
result_seq_AIAR = testagent(sequence_AIAR, testenvs_AIA)
println("Exp. sequence AIAR: ", result_seq_AIAR[6])
#--------------------------------------------------------------------------------------------------------------
# Priority Rule SPT
#--------------------------------------------------------------------------------------------------------------
# Priority Rule SPT AIM
SPT_AIM = createagent(createRule(Priorityfunction("TT")),"AIM")
result_rule_AIM = testagent(SPT_AIM,testenvs_AIM)
println("SPT AIM: ", result_rule_AIM[6])

# Priority Rule SPT AIA
SPT_AIA = createagent(createRule(Priorityfunction("TT")),"AIA")
result_rule_AIA = testagent(SPT_AIM,testenvs_AIA)
println("SPT AIA: ", result_rule_AIA[6])

#--------------------------------------------------------------------------------------------------------------
# Plots
#--------------------------------------------------------------------------------------------------------------

# boxplot results
boxplot([result_seq_AIA[10], result_seq_AIAR[10], result_rule_AIM[10], result_rule_AIA[10], results_GP_AIM_cycling[10]],
        label = ["Exp. AIA" "Exp. AIAR" "SPT AIM" "SPT AIA" "GP AIM cycling"],
        title = "Boxplot of gaps",
        ylabel = "gap",
        xlabel = "model")
