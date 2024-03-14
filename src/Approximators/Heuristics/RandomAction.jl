struct RandomAction <: AbstractApproximator
    trainable::Bool
    rng::AbstractRNG
end

function createRandomAction(; rng = Random.default_rng())
    RandomAction(false, rng)
end

function test(a::Agent{RandomAction},env,nrseeds)
    evalrandomaction(a, env, a.objective, nrseeds, a.rng)
end

function evalrandomaction(agent, env, objective, nrsamples,rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    for i in 1:nrsamples
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = singleinstancerandom(agent, env, objective, pointer, rng)
        append!(metrics, tmpmetrics)
        append!(fitness, tmpfitness)
        if env.type == "usecase"
            if objective == [-1.0,-0.0,-0.0,-0.0,-0.0,-0.0]
                piobjective = env.samples[pointer]["objective_[1, 0]"]
                append!(gaps, (tmpfitness/piobjective) -1)
            else
                piobjective = env.samples[pointer]["objective_[0, 1]"]
                append!(gaps, (tmpfitness - piobjective))
            end
        else
            piobjective = env.samples[pointer]["objective"]
            append!(gaps, (tmpfitness/piobjective) -1)
        end
    end

    return sum(fitness),fitness, gaps

end

function singleinstancerandom(agent, env, objective, pointer, rng)
    actionspace = agent.actionspace
    
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    counterdict = Dict(m => 1 for m in env.problem.machines)
    t = false
    while !t
        action = actionrandom(env, actionspace, rng)
        # println("action:", env.siminfo.mappingjobs[action[1]]," ", env.siminfo.mappingmachines[action[2]]," ", action[3])
        _, rewards, t, _ = step!(env, action, rng)
        metrics += rewards
    end

    fitness = dot(objective,metrics)
    return fitness, metrics
end

function actionrandom(env, actionspace, rng)
    actions = findall(x->x == 1, env.state.actionmask)
    index = rand(rng, 1:length(actions))
    job = actions[index][1]
    m = actions[index][2]

    if actionspace == "AIM"
        return (job ,m, "S")
    
    elseif env.state.executable[job]
        if o === nothing
            return (job, m, "S")
        else
            #println("because machine is occupied")
            return (job, m, "W")
        end
    else
        if actionspace == "AIAR"
            if env.state.setup[m] !== nothing && env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"] !== 0
                return (job, m, "R")
            else
                return (job, m, "W")
            end
        else
            return (job, m, "W")
        end
    end    
end