abstract type AbstractApproximator end
"""
mutable struct of an Agent 
"""
mutable struct Agent{A<:AbstractApproximator}
    "defines the type of the agent, i.e. GP, DRL, or ParaPrio"
    type::String
    "action space the agent uses to train"
    actionspace::String
    "mask that is used additionally"
    actionmask::Array{String}
    "task the agent is trained, i.e. tailored or generalized"
    purpose::String
    "will store the problem it was assigned to as string"
    problemInstance # placeholder: problem was solved
    "will store the envs used for training"
    trainingenvs::Union{Vector{Env},Nothing} # placeholder: problem was solved
    "objective the agent is trained for,"
    "[makespan, totalFlowTime, tardiness, discretetardiness, numberTardyJobs, numSetups]" #TODO length of input assert??
    objective::Vector{Float64}
    "stores the struct of the intialized agent based on previous inputs"
    approximator::A # initialized structure for the agent based on settings
    "stores the trained model"
    model # placeholder: model
    rng::AbstractRNG
    logger::Dict{String,Vector{Any}}
    
end

include("Approximators/Approximators.jl")

"""
    create_agent(type::String, actionspace:: String; purpose::String = "tailored", obj = [1.0,0.0,0.0,0.0,0.0,0.0], settings::Dict = Dict())

creates the agent and returns it as a struct
"""
function createagent(approximator::AbstractApproximator, actionspace::String; actionmask = ["none"], purpose::String = "tailored", obj = [1.0,0.0,0.0,0.0,0.0,0.0])
    if approximator.trainable
        setobj!(approximator,obj)
        envs = approximator.envs # TODO
        instancename = envs[1].name
    else
        envs = nothing
        instancename = nothing
    end
    logging = Dict("reward" => [])
    type = string(typeof(approximator))[14:end]

    Agent(type, actionspace, actionmask, purpose, instancename , envs, obj, approximator, nothing,
    type == "E2E" || type == "WPF" ? approximator.policy.rng : approximator.rng, logging)
end

"""
    update_agent_settings!(agent::Agent; settings::Dict = Dict(), obj= [1.0,0.0,0.0,0.0,0.0,0.0])

    change settings or objective of agent
    resets model / might be updated to keep model -> test effect of keeping model for other purposes

"""
function updateagentsettings!(agent::Agent; settings::Dict = Dict(), obj= [1.0,0.0,0.0,0.0,0.0,0.0])

    for c in keys(settings)
        if haskey(agent.settings,c)
            agent.settings[c] = settings[c]
        end
    end

    agent.objective = obj
    agent.approximator = initAgent(agent.type, agent.trainingenvs, obj, agent.settings)
    
    # TODO store intermediate result! change in order to keep model and only train further?
    agent.model = nothing

    println("Agent's settings are updated")
end
"""
   create_agent(file::String)

create agent instead from a file -> trained model
"""
function createagent(file::String)
    #
    #
    # TODO
    #
    #
    #
    return Agent
end
"""
    trainagent!(agent::Agent; generations=100, evalevery = 0, finaleval = true, kwargs...)

    evalevery: how often to evaluate the agent, 0 => no eval while training

traines the agent and changes its model variable  
"""
function trainagent!(agent::Agent; generations=100, evalevery = 0, finaleval = true, testenvs = nothing, kwargs...)
    # training loop call function based on type of agent / i.e. approximator
    if finaleval || evalevery > 0
        @assert testenvs !== nothing "No testenvs provided."
    end
    @assert length(agent.trainingenvs) > 0 "No environment set to train on. call setenvs! on Agent first"

    # tmpT = testenvs[1].samples
    # println(tmpT)
    # @assert tmpT[1]["setup_time"][5][5][16] == 3.0
    # @assert setupsample(agent.approximator.envs[1].problem, 16, 5, 5, agent.rng) == 3.0
    # @assert tmpT[1]["processing_time"][5][4][11] == 46.0
    # @assert processsample(agent.approximator.envs[1].problem, 5, 4, 11, agent.rng) == 46.0
    # error()

    train!(agent, generations, evalevery, testenvs, finaleval; kwargs...)
end

"""
    export_agent(agent::Agent, target::String; kwargs)

exports the agent and allows it to be stored in the target location  
"""
function exportagent(agent::Agent, name::String, target::String; kwargs...)

end

"""
    show_settings(agent::Agent)

    prints the settigns of the agent to the console  
"""
function showsettings(agent::Agent)
    if agent.model !== nothing
        s = "(trained)"
    else
        s ="(untrained)"
    end
    
    @printf("\n Agent %s general settings are: ", s)
    println()

    df_stats = DataFrame(Setting=String[],Value=Any[])
    push!(df_stats, ("Type", agent.type))
    push!(df_stats, ("Action Space", agent.actionspace))
    push!(df_stats, ("Purpose", agent.purpose))
    push!(df_stats, ("Objective", agent.objective))
    push!(df_stats, ("problemInstance", agent.problemInstance))
    push!(df_stats, ("different envs ", length(agent.trainingenvs)))

    pretty_table(df_stats,alignment=:c)

    println("Agent specific settings are:")

    pretty_table(agent.settings,alignment=:c)
    
end

"""
    setenvs!(agent::Agent, envs::Vector{Env})

    set the env of an agent to train on
    if existing already replaces them
    propagated to algorithm envs
"""
function setenvs!(agent::Agent, envs::Vector{Env})
    for env in envs
        env.actionspace = agent.actionspace
    end
    # TODO for NN based agents add check if apdding is active or not and statesize/actionsize vs env required
    agent.trainingenvs = envs
    if length(agent.approximator.envs) >0
        for _ in eachindex(agent.approximator.envs)
            deleteat!(agent.approximator.envs,1)
        end
        append!(agent.approximator.envs,envs)
    else
        append!(agent.approximator.envs,envs)
    end
end

function setobj!(agent::Agent, obj::Vector{Float64})
    agent.objective = obj
    if length(agent.approximator.objective) >0
        for _ in eachindex(agent.approximator.objective)
            deleteat!(agent.approximator.objective,1)
        end
        append!(agent.approximator.objective,obj)
    else
        append!(agent.approximator.objective,obj)
    end
end

function generatetestset(envstring::Vector{Dict{String,String}},nrseeds; actionspace = "AIM", actionmask = ["none"], deterministic = false)
    @assert nrseeds <= 100 "only 100 sample files do exist"
    envs::Vector{Env} = []
    for e in envstring
        push!(envs,createenv(fromfile = true, as = actionspace, am = actionmask, nSamples = nrseeds,
                                layout = e["layout"] ,instanceType = e["instancetype"],
                                instancesettings = e["instancesettings"], datatype = e["datatype"] ,instanceName = e["instancename"], deterministic = deterministic
        ))
        # TODO read deterministic PI solutions and store them in dict (match to postion in env and position of sample!)
    end
    return envs
end

"""
    testagent(agent::Agent,envs::Vector{String},nrseeds)


    test trained agent with envs. each env needs to be created with the samples to evaluate!
"""
function testagent(agent::Agent, testenvs) #TODO or only add Json files and envs are generated within function

    # absolute metrics
    totalobjVal = 0
    envobj = []
    envseedobj = []
    pisolenv = []

    # relativ metrics
    avggap = 0
    avggapenv = []
    worstgap = 0
    worstgapenv = []
    gaps = []

    insamplevalue = agent.type == "GP" ? agent.model[1][2] : 0.0 # TODO for all
    nrseeds = length(testenvs[1].samples)
    for env in testenvs
        tmpO,tmpS,tmpG = test(agent,env,nrseeds) # returns total env objective, objective for each seed and the gap of each seed
        totalobjVal += tmpO
        append!(gaps, tmpG)
        append!(worstgapenv, maximum(tmpG))
        append!(avggapenv,mean(tmpG))
        push!(envseedobj,tmpS)
        if env.type == "usecase"
            if agent.objective == [1.0,0.0,0.0,0.0,0.0,0.0]
                push!(pisolenv, [v["objective_[1, 0]"] for v in values(env.samples)])
            else
                push!(pisolenv, [v["objective_[0, 1]"] for v in values(env.samples)])
            end
        else
            push!(pisolenv, [v["objective"] for v in values(env.samples)])
        end
        tmpenvobj = tmpO/nrseeds
        append!(envobj,tmpenvobj)
    end
    avggap = mean(avggapenv)
    worstgap = maximum(worstgapenv)
    totalobjVal /= (nrseeds*length(testenvs))

    losstoinsample = (totalobjVal - insamplevalue/totalobjVal) - 1
    
    return totalobjVal, envobj, envseedobj, pisolenv, losstoinsample, avggap, avggapenv, worstgap, worstgapenv, gaps
end


function creategantt(agent::Agent,envstring,seed)
    env = createenv(fromfile = true, specificseed = seed, as = agent.actionspace, am = agent.actionmask, nSamples = 1,
                                layout = envstring["layout"] ,instanceType = envstring["instancetype"],
                                instancesettings = envstring["instancesettings"],
                                datatype = envstring["datatype"] ,instanceName = envstring["instancename"])


    assignments = DataFrames.DataFrame()
    sequence = Dict(m => [] for m in env.problem.machines) #TODO use / store? or move to normal testing and store all and compare?
    # TODO define individuum, store intraiing performance as well!
    t = false
    setsamplepointer!(env.siminfo,1)
    while !t
        action = nextaction(agent,env)

        if action[3] == "R"
            append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string("Setup ",action[1]),
            Start = env.siminfo.time, Machine = action[2],
            End = (env.siminfo.time + env.samples[1]["setup_time"][action[2]][env.state.setup[action[2]]][action[1]])))

        elseif action[3] == "S"
            push!(sequence[action[2]],action[1])
            operation = env.problem.orders[action[1]].ops[env.siminfo.pointercurrentop[action[1]]]
            if env.state.setup[action[2]] === nothing || env.problem.setupstatejob[action[2]][env.state.setup[action[2]]][action[1]]["mean"] == 0
                append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string(action[1]),
                Start = env.siminfo.time, Machine = action[2],
                End = (env.siminfo.time + env.samples[1]["processing_time"][action[1]][operation][action[2]])))
            else
                append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string("Setup ",action[1]),
                Start = env.siminfo.time, Machine = action[2],
                End = (env.siminfo.time + env.samples[1]["setup_time"][action[2]][env.state.setup[action[2]]][action[1]])))
                append!(assignments, DataFrames.DataFrame(Job = action[1], Type = string(action[1]),
                Start = (env.siminfo.time + env.samples[1]["setup_time"][action[2]][env.state.setup[action[2]]][action[1]]), Machine = action[2],
                End = (env.siminfo.time + env.samples[1]["setup_time"][action[2]][env.state.setup[action[2]]][action[1]]
                + env.samples[1]["processing_time"][action[1]][operation][action[2]])))
            end
        end
        state, reward, t, info = step!(env,action, agent.rng)
    end
    # setupnames = [string("Setup ", x) for x in keys(env.problem.orders)]
    # processnames = [string(x) for x in keys(env.problem.orders)]
    # colDomain = [setupnames, processnames]
    # colRange = ["#e08c8c", "#dc0202", "#fff998", "#fff000", "#80e78e", "#00ad18", "#85a5ee", "#004dff"]
   
    p = assignments |> @vlplot(
        mark = {
            type = :bar,
            cornerRadius = 10
        },
        width = 2000,
        height = 1000,
        y="Machine:n",
        x="Start:q",
        x2="End:q",
        color={
            field="Type",
            type = "nominal",
            #scale = {domain = colDomain, range = colRange}
        } 
    ) 
    +
    @vlplot(
        mark={:text,align = "center",x="Start", y="Machine", dx=20, dy=10},
        text="Job",
        detail={aggregate="count",type="qualitative"}
    )
    # + 
    # @vlplot(
    #     mark={:text,align = "center",x="Start", y="Machine", dy=-10},
    #     text="Due"
    # )
    # VegaLite.save(string(@__DIR__,"/Plots/",instanceName,"_",SIM_METHOD, "_gantt.pdf"), p)
    display(p)
end