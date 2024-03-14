using Revise
using SchedulingRL
using Test

@testset "SchedulingRL.jl" begin
    # Write your tests here.
    #-----------------------------------------------------------------------
    # basic functionality
    #-----------------------------------------------------------------------
    # Env = createenv(as = "AIM",instanceName = "hhhhhh-0_ii_Expo5")
    # Env = createenv(as = "AIM", instanceType = "usecase", instanceName = "useCase_2_stages")
    # Env = createenv(as = "AIM")
    # @test Env.state.setup[1] === nothing
    # setsamples!(Env)
    # @test haskey(Env.samples,1) == true 
    # a = copy(Env.samples[1])
    # setsamples!(Env)
    # @test Env.samples[1] != a
    # @test Env.samples[1] != a
    
    # AgentGP = createagent(createGP(), "AIM")
    # @test Agent.type == "GP"
    # @test Agent.trainingenvs == []
    # setenvs!(AgentGP, [Env])
    # @test length(Agent.trainingenvs) == 1
    # setenvs!(Agent, [Env])
    # @test length(Agent.approximator.envs) == 1
    
    # println("start of training")
    # @time trainagent!(AgentGP,generations = 4)

    # @test Agent.model !== nothing

    # testenvs = [Dict("layout" => "Flow_shops" ,
    #                 "instancetype" => "benchmark" ,
    #                 "instancesettings" => "base_setting",
    #                 "datatype" => "data_ii",
    #                 "instancename" =>"llllll-0_ii_Expo5")]
    
    # a = testagent(AgentGP,testenvs,100)
    
    # println("Avg obj value is: ", a[1])
    # println("Avg obj values per instance are: \n", a[2])
    # println("Objective values are: \n", a[3])
    # println("PI objective values are: \n", a[4])
    # println("Variance insample outofsample is: \n", a[5])
    # println("The average PI gap is: ", a[6])
    # println("The average PI gaps per instance are: \n", a[7])
    # println("The worst gap is: ", a[8])
    # println("The worst gaps per instance are: \n", a[9])
    # println("All gaps are: \n", a[10])

    # creategantt(AgentGP,testenvs[1],2)
    
    #-----------------------------------------------------------------------
    # Speed test with random actions
    #-----------------------------------------------------------------------
    # t = false
    # totalreward = [0,0,0,0,0,0]
    # @time for _ in 1:1000
    #     while !t
    #         actions = findall(x->x == 1, Env.state.actionmask)
    #         index = rand(1:length(actions))
    #         action = (actions[index][1],actions[index][2])
    #         state, reward, t, info = step!(Env,action)
    #         totalreward += reward
    #     end
    #     resetenv!(Env)
    #     t = false
    # end
    
    # becnhmark
    
    testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>"lhmhlh-0_ii_Expo5")]

    BenchmarkSPT = createagent(createRule(Priorityfunction("0-TT")),"AIM")
    resultrule = testagent(BenchmarkSPT,testenvs,100)
    println("The average PI gap is: ", resultrule[6])
    println("The worst gap is: ", resultrule[8])
    println("All gaps are: \n", resultrule[10])


    #-----------------------------------------------------------------------
    # test GP agent
    #-----------------------------------------------------------------------
    println("start")

    Env = createenv(as = "AIM", instanceName = "lhmhlh-0_ii_Expo5")
    println("env generated")

    AgentGP = createagent(createGP([Env]), "AIM")
    
    println("start training")
    @time trainagent!(AgentGP,generations = 4)
    testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>"lhmhlh-0_ii_Expo5")]
    a = testagent(AgentGP,testenvs,100)
    
    println("The average PI gap is: ", a[6])
    println("The worst gap is: ", a[8])
    println("All gaps are: \n", a[10])

    #-----------------------------------------------------------------------
    # test E2E agent
    #-----------------------------------------------------------------------

    using Flux
    using ParameterSchedulers

    Env = createenv(as = "AIM", instanceName = "lhmhlh-0_ii_Expo5")
    model = Chain(Dense(statesize(Env) => 30, selu),
                Dense(30 => 30, selu),)
    AgentE2E = createagent(
        createE2E( 
                ActorCritic(
                    actor = Chain(model,Dense(30 =>actionsize(Env))), # no specified activation function = linear activation
                    critic = Chain(model,Dense(30 => 1))),            # no specified activation function = linear activation
                    [Env],
                    [],
                    lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2))
                )
            , "AIM")
            
    setenvs!(AgentE2E,[Env])

    @time trainagent!(AgentE2E,generations = 100)


    testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>"lhmhlh-0_ii_Expo5")]
    
    a = testagent(AgentE2E,testenvs,100)
    println("The average PI gap is: ", a[6])
    println("The worst gap is: ", a[8])
    println("All gaps are: \n", a[10])


    #-----------------------------------------------------------------------
    # test WPF Agent
    #-----------------------------------------------------------------------
    
    Env = createenv(as = "AIM", instanceName = "lhmhlh-0_ii_Expo5")
    model = Chain(Dense(statesize(Env) => 30, selu),
                Dense(30 => 30, selu),)

    prio = Priorityfunction("PT + DD + RO + RW + ON + JS + RF + CT + EW + JWM + ST + NI + NW + SLA + CW + CJW + TT")
    # prio = Priorityfunction("PT + ON") to test if arbitrary functions work


    output = numberweights(prio)
    AgentWPF = createagent(
        createWPF( 
                ActorCritic(
                    actor = GaussianNetwork(
                        pre = Chain(model,Dense(30 => 30)),
                        μ = Chain(Dense(30, output, tanh)),
                        σ = Chain(Dense(30, output, sigmoid)),
                        ),
                    critic = Chain(model,Dense(30 => 1))),
                    [Env],
                    [],
                    prio,
                    masked = false,
                    lr_scheduler = ParameterSchedulers.Stateful(Inv(λ = 1e-1, γ = 0.2, p = 2))
                )
            , "AIM")

    setenvs!(AgentWPF,[Env])
    @time trainagent!(AgentWPF,generations = 100)

    testenvs = [Dict("layout" => "Flow_shops" ,
                    "instancetype" => "benchmark" ,
                    "instancesettings" => "base_setting",
                    "datatype" => "data_ii",
                    "instancename" =>"lhmhlh-0_ii_Expo5")]
    
    a = testagent(AgentWPF,testenvs,100)
    println("The average PI gap is: ", a[6])
    println("The worst gap is: ", a[8])
    println("All gaps are: \n", a[10])

    # creategantt(AgentWPF,testenvs[1],2)

end
