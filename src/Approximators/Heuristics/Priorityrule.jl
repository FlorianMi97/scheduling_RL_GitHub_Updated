
struct Priorityrule <: AbstractApproximator
    pf::Priorityfunction
    trainable::Bool
    rng::AbstractRNG
end

function createRule(prio; rng = Random.default_rng())
    Priorityrule(prio,false, rng) # TODO multiple envs...
end

function test(a::Agent{Priorityrule},env,nrseeds)
    Benchmark_rule(a.approximator.pf, env, a.objective, nrseeds, a.rng)
end

function Benchmark_rule(prio, env, objective, nrsamples,rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    for i in 1:nrsamples
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = singleinstance(prio, env, objective, pointer, rng)
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

function singleinstance(priorule::Priorityfunction, env, objective, pointer, rng)
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    t = false
    while !t
        actionindex =  [1 for _ in 1:numberweights(priorule)]
        action = translateaction([priorule], [actionindex], env )
        _, rewards, t, _ = step!(env, action, rng)

        metrics += rewards
    end

    fitness = dot(objective,metrics)
    return fitness, metrics
end