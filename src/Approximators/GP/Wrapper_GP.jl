
"""
    generates actions based on the individuum and the current state
    type allows to select actions for each machine either 
        "greedy" -> repetitively selecting the best action based on priority
        "optimized" -> selecting the best set of actions based on priority
"""
function actionsfromindividuum(individuum, env; type="greedy")
    actionvector = [] # define how actions are encoded: (job,machine,type)
    pa = [Tuple(x) for x in findall(x -> x == 1, env.state.actionmask)] # possible actions

    # println("possible actions are: ", pa)
    # generate matrix
    jobs = unique([x[1] for x in pa])
    machines = unique([x[2] for x in pa])

    # println("jobs: ",jobs)
    # println("machines: ",machines)

    priomatrix = fill(Inf,(length(jobs),length(machines)))
    #TODO baseline and reduced tree!!!
    # for (k,t) in Individuums
    #     if k == "baseline"
    # syms=[symbols(ex) for ex in t]

    # generate required symbols to calculate priority individuums
    syms=[symbols(ex) for ex in individuum]
    # define feature functions 
    featurefunctions = Dict(:PT => getPT, :DD => getDD, :RO => getRO, :RW => getRW, :ON => getON, :JS => getJS,
                        :RF => getRF, :CT => getCT, :EW => getEW, :JWM => getJWM, :ST => getST, :NI => getNI, :NW => getNW, :SLA => getSLA,
                        :CW => getCW, :CJW => getCJW, :TT => getTT, :BSA => getBSA, :DBA => getDBA)

    # fill matrix
    for ix in CartesianIndices(priomatrix)
        j,m = Tuple(ix)
        if (jobs[j],machines[m]) in pa
            # println("job: ", jobs[j] ," machine: ", machines[m])
            priomatrix[j,m] = calcprio(individuum, env, jobs[j], machines[m], syms, featurefunctions) 
            #TODO apply individuum for job, machine // specify fucntion ! additional parameters?
        end
    end
    # println("matrix is: ", priomatrix)
    if type == "greedy"

        # select pair
        prio, sa = findmin(priomatrix) 
        
        # println("choosen indices are: ",sa)
        # translate pair to action tuple
        sa = (jobs[sa[1]], machines[sa[2]])

        type = ""


        # if waiting -> nothing
        if env.state.occupation[sa[2]] !== nothing
            type = "W"
        elseif env.state.executable[sa[1]] == false
            # if action design is allowing it perform risk setup
            if env.actionspace == "AIAR"
                if env.state.setup[sa[2]] !== nothing && env.problem.setupstatejob[sa[2]][env.state.setup[sa[2]]][sa[1]]["mean"] !== 0
                    #println(env.problem.setupstatejob[action[2]][env.state.setup[action[2]]][action[1]])
                    type = "R"
                    #println("risksetup")
                else
                    type = "W"
                    #println("waiting")
                end
            # else waiting
            else
                type = "W"
                #println("waiting")
            end
        # both ready -> schedule
        else # same as: if env.state.occupation[m] === nothing && env.state.executable[j] == true
            type = "S"
            #println("schedule")
        end
        action = (sa[1],sa[2],type)
        push!(actionvector, action)
    elseif type == "optimized"

        # TODO 
        # optimized total priority over vector of pairs but no update -> feasible set of actions needed!!!

        # translate all pairs to vector of action tuples
    end
    # println("selected action is: ", actionvector[1])
    return actionvector #TODO return prio as well? # TODO stay by single action? -> remove vector!
end

# TODO !!!
function calcprio(individuum, env, job, machine, syms, featurefunctions)
    if env.actionspace != "AIM"
        st = getST(env,job,machine,0,0)
        t = getCT(env,job,machine,0,0)
        waiting = max(env.state.expfinish[machine][2] - t,0)
        idle = defineidle(env, job,t)
        idle -= max(min((idle - waiting), st),0)
        @assert idle >= 0
        
    else
        waiting = 0
        idle = 0
    end

    vals = Dict(f => featurefunctions[f](env,job,machine, idle, waiting) for f in syms[1]) # syms[stage])# TODO calcualte all features based on state, job, machine
    prio = evaluate(individuum[1], vals)
    return prio
end


function evalfitnesssample(individuum, env, objective, type, pointer, rng)
    metrics = [0,0,0,0,0,0]

    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    t = false
    if type == "greedy"
        while !t
            action = actionsfromindividuum(individuum,env)[1]
            state, reward, t, info = step!(env,action,rng)
            metrics += reward
        end
    else
        while !t
            # TODO not implemented yet! 
            action = actionsfromindividuum(individuum,env,type = type)
            for a in action
                state, reward, t, info = step!(env,a,rng)
                metrics += reward
            end
        end
    end

    fitness = dot(-objective,metrics)
    return fitness, metrics
end

# TODO function to get average fitness over multiple samples // iterate over samples
"""
evalfitness(individuum, env, objective, nrsamples,type) -> Float64,Vector{Float64},Matrix{Float64}

type: greedy or optimized

"""
function evalfitness(individuum, env, objective, nrsamples, rng, sample_tens_digit; type = "greedy")
    metrics = []
    fitness = []
    #TODO track individual performance of samples as well? NSGA-II ?
    #TODO track individual metrics as well? NSGA-II ?

    for i in 1:nrsamples

        #TODO logic to define if use samples or sample new!  
        pointer = isempty(env.samples) ? nothing : (i + nrsamples * sample_tens_digit)
        tmpfitness, tmpmetrics = evalfitnesssample(individuum, env, objective, type, pointer, rng)
        append!(metrics, tmpmetrics)
        append!(fitness, tmpfitness)
    end

    return sum(fitness),fitness, metrics
end

function testfitness(individuum, env, objective, nrsamples, rng; type = "greedy")
    metrics = []
    fitness = []
    gaps = []

    for i in 1:nrsamples
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = evalfitnesssample(individuum, env, objective, type, pointer, rng)
        append!(metrics, tmpmetrics)
        append!(fitness, tmpfitness)
        if env.type == "usecase"
            if objective == [1.0,0.0,0.0,0.0,0.0,0.0]
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

#-----------------------------------------------------------------------------------
# FEATURE HELPER FUNCTIONS
#------------------------------------------------------------------------------------
function defineidle(env, job,t)
    if env.state.executable[job]
        idle = 0
    else
        idle = env.siminfo.pointercurrentop[job] == 1 ? 0 : max(env.state.expfinish[findfirst(i -> env.state.occupation[i] == job,
                                    env.problem.orders[job].eligmachines[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]-1]])][2] - t,0)
    
    end
    return idle
end


function getDD(env,job,machine, idle, waiting)
    return env.state.generalfeatures["DD"][job]
end

function getRO(env,job,machine, idle, waiting)
    return env.state.generalfeatures["RO"][job]
end

function getRW(env,job,machine, idle, waiting)
    return env.state.generalfeatures["RW"][job]
end


function getON(env,job,machine, idle, waiting)
    return env.state.generalfeatures["ON"][job]
end

function getJS(env,job,machine, idle, waiting)
    return env.state.generalfeatures["JS"][job]
end

function getRF(env,job,machine, idle, waiting)
    return env.state.generalfeatures["RF"][job]
end

function getEW(env,job,machine, idle, waiting)
    return env.state.generalfeatures["EW"][machine]
end

function getCW(env,job,machine, idle, waiting)
    return env.state.generalfeatures["CW"][machine]
end

function getJWM(env,job,machine, idle, waiting)
    return env.state.generalfeatures["JWM"][machine]
end

function getCJW(env,job,machine, idle, waiting)
    return env.state.generalfeatures["CJW"][machine]
end

function getPT(env,job,machine, idle, waiting)
    return env.problem.orders[job].processingtimes[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]]][machine]["mean"]
end

function getTT(env,job,machine, idle, waiting)
    return getPT(env,job,machine, idle, waiting) + getST(env,job,machine, idle, waiting) + max(idle, waiting)
end

function getST(env,job,machine, idle, waiting)
    return env.problem.setupstatejob[machine][env.state.setup[machine]][job]["mean"]
end

function getNI(env,job,machine, idle, waiting)
    return idle
end

function getNW(env,job,machine, idle, waiting)
    return waiting
end

function getSLA(env,job,machine, idle, waiting)
    return env.problem.setupstatejob[machine][env.state.setup[machine]][job]["mean"] !== 0 &&
                            any(env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"] == 0
                            for m in env.problem.orders[job].eligmachines[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]]]) ? 1 : 0  
end

function getBSA(env,job,machine, idle, waiting)
    setup = env.problem.setupstatejob[machine][env.state.setup[machine]][job]["mean"]
    for m in env.problem.orders[job].eligmachines[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]]]
        if env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"] < setup
            return 1
        end
    end
    return 0  
end

function getDBA(env,job,machine, idle, waiting)
    setup = env.problem.setupstatejob[machine][env.state.setup[machine]][job]["mean"]
    best = setup
    for m in env.problem.orders[job].eligmachines[env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]]]
        if env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"] < setup
            best = env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"]
        end
    end
    difference = setup - best
    return difference
end

function getCT(env,job,machine, idle, waiting)
    return env.state.generalfeatures["CT"]
end