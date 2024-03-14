struct ExpSequence <: AbstractApproximator
    sequence::Dict
    trainable::Bool
    rng::AbstractRNG
    actionspace::String
end

function createExpSequence(instance, env, actionspace; rng = Random.default_rng())
    if actionspace == "AIM" warn("AIM is not compatible with Exp. Sequence. This might cause errors") end
    sequence = Dict()
    dir =  pathof(SchedulingRL)|> dirname |> dirname

    if contains(instance, "useCase")
        if actionspace == "AIAR"
            file = string(dir,"/results/Flow_shops/Use_Case_Cereals/base_setting/decoupled_setups/sequence_", instance, "_[0, 1].json")
        else
            file = string(dir,"/results/Flow_shops/Use_Case_Cereals/base_setting/coupled_setups/sequence_", instance, "_[0, 1].json")
        end

    else
        if actionspace == "AIAR"
            file = string(dir,"/results/Flow_shops/Benchmark_Kurz/base_setting/results_ii/decoupled_setups/sequence_", instance, ".json")
        else
            file = string(dir,"/results/Flow_shops/Benchmark_Kurz/base_setting/results_ii/coupled_setups/sequence_", instance, ".json")
        end
    
    end
    open(file, "r") do f
        sequence = JSON.parse(f)
    end
    sequence = convertsequence(sequence, env)

    ExpSequence(sequence, false, rng, actionspace) # TODO multiple envs...
end

function test(a::Agent{ExpSequence},env,nrseeds)
    evalfixseq(a.approximator, env, a.objective, nrseeds, a.rng)
end

function evalfixseq(approx, env, objective, nrsamples,rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    for i in 1:nrsamples
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = singleinstance(approx, env, objective, pointer, rng)
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

function singleinstance(approx, env, objective, pointer, rng)
    sequence = approx.sequence
    actionspace = approx.actionspace
    
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    counterdict = Dict(m => 1 for m in env.problem.machines)
    t = false
    while !t
        action = actionfromsequence!(counterdict, sequence, env, actionspace)
        # println("action:", env.siminfo.mappingjobs[action[1]]," ", env.siminfo.mappingmachines[action[2]]," ", action[3])
        _, rewards, t, _ = step!(env, action, rng)

        time = env.siminfo.time
        # println("------------------------- $time -------------------------")
        #println("nextevent: ", env.siminfo.nextevent)
        metrics += rewards
    end

    fitness = dot(objective,metrics)
    return fitness, metrics
end

function actionfromsequence!(counterdict, sequence, env, actionspace)
    for (m,o) in env.state.occupation
        # println(env.state.actionmask[:,m])
        # println(env.state.occupation[m])
        if sum(env.state.actionmask[:,m]) > 0 # otherwise its occupied
            if counterdict[m] > length(sequence[m])
                continue # since machine already finished all its jobs 
            else
                job = sequence[m][counterdict[m]]
                #println(env.state.executable[job])
                if env.state.actionmask[job,m] != 1
                    # println("order next op is: ", env.siminfo.pointercurrentop[job])
                    # println("machine: ", m)
                    # println("the machines that are able to process ne next op are: ", env.problem.orders[job].eligmachines[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]]])
                    # println("action ", env.siminfo.mappingjobs[job], " ", env.siminfo.mappingmachines[m] , " not allowed yet")
                    continue # since action is not allowed yet
                elseif env.state.executable[job]
                    if o === nothing
                        counterdict[m] += 1
                        return (job,m,"S")
                    else
                        #println("because machine is occupied")
                        return (job,m,"W")
                    end
                else
                    if actionspace == "AIAR"
                        if env.state.setup[m] !== nothing && env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"] !== 0
                            return (job,m,"R")
                        else
                            return (job,m,"W")
                        end
                    else
                        return (job,m,"W")
                    end
                end
            end
        end
    end
    # found no action to act on -> select waiting for any possible action
    cart = findfirst(x -> x == 1, env.state.actionmask)
    return (cart[1],cart[2],"W")
    
end


function convertsequence(sequence, env)
    converted = Dict()

    mapmachineindex = Dict(m => i for (i,m) in env.siminfo.mappingmachines)
    mapjobindex = Dict(j => i for (i,j) in env.siminfo.mappingjobs)
    #convert machines to ints
    for (m, jobs) in sequence
        machine = mapmachineindex[m]
        tmpList = []
        for job in jobs
            append!(tmpList, mapjobindex[job])
        end
        converted[machine] = tmpList
    end
    return converted
end