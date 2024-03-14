
##################### Helper functions for simulation ###################
"""
finds the next event that has no flag
used as function f ∈ findmin()
"""
function flagmin(a)
    if a[2] != 0 #|| a[1] < a[3] # TODO if a[3] not needed, delete arrayzip
        return Inf
    else
        return a[1]
    end
end

"""
zips every entry of an array with a constant
"""
function arrayzip(array,c) # TODO delete arrayzip if a[3] not needed
    return [[a;c] for a ∈ array]
end

"""
    returns if a machine requires a decision
"""
function macdecne(env::Env)
    # 
    for m ∈ eachindex(env.siminfo.nextevent)
        if  env.state.occupation[m] === nothing && sum(env.state.actionmask[:,m]) > 0
            return true
        end
    end
    return false
end

##########################################################################
function waitingaction!(env::Env, action::Tuple{Int,Int,String})
    # set tabu flag on this machine's next event -> prevent deadlock
    if env.state.occupation[action[2]] === nothing
        env.siminfo.nextevent[action[2]][2] = 1 # set flag for this
    end 
    updatedactionmask!(env, action)
end

function risksetupaction!(env::Env, action::Tuple{Int,Int,String},rng::AbstractRNG)
    # take value from pre-generated samples
    job = action[1]
    machine = action[2]
    setup = env.state.setup[machine]
    sample = env.siminfo.pointersample
    if sample !== nothing
        env.siminfo.nextevent[machine][1] =   (env.siminfo.time
                                                + env.samples[sample]["setup_time"][machine][setup][job])
    else
    # new sample based on distribution
        env.siminfo.nextevent[machine][1] =   (env.siminfo.time
                                                + setupsample(env.problem,job,
                                                                machine,
                                                                setup,
                                                                rng)) 
    end

    env.siminfo.activitytimes[machine] = [[env.siminfo.time + env.problem.setupstatejob[machine][setup][job]["min"], 0.0, 0.0], # store min time
                                            [env.deterministic ? env.problem.setupstatejob[machine][setup][job]["mean"] : mean(env.problem.setupstatejob[machine][setup][job]["dist"]), 0], # store mean of stochastic part time
                                            [env.siminfo.time + env.siminfo.nextevent[machine][1],0], # store real time for faster comparison later
                                            env.deterministic ? ["Determinisitic", "None"] : [env.problem.setupstatejob[machine][setup][job]["type"], "None"],
                                            [env.problem.setupstatejob[machine][setup][job]["max"],0]
                                        ]
    
    env.state.starttime[machine] = env.siminfo.time
    env.state.expfinish[machine] = [env.siminfo.activitytimes[machine][1][1],
                                    env.siminfo.activitytimes[machine][1][1] + env.siminfo.activitytimes[machine][2][1]]
    
    env.siminfo.numsetups += 1

    for j ∈ env.siminfo.notfinished
        # updated based on changed setup!
        if env.siminfo.pointercurrentop[j] < length(env.problem.orders[j].ops)
            if machine ∈ env.problem.orders[j].eligmachines[env.problem.orders[j].ops[env.siminfo.pointercurrentop[j]]]
                env.state.generalfeatures["ON"][j] += (env.problem.setupstatejob[machine][env.problem.orders[job].type][j]["mean"] # + new setup time
                                                    - env.problem.setupstatejob[machine][setup][j]["mean"]) # - old setup time
            end
        end
    end
    
    # change state
    updatestate!(env, action)

    updatedactionmask!(env, action)
end

function scheduleaction!(env::Env, action::Tuple{Int,Int, String},rng::AbstractRNG)
    job = action[1]
    machine = action[2]
    setup = env.state.setup[machine]
    operation = env.problem.orders[job].ops[env.siminfo.pointercurrentop[job]] # TODO check if it works? and what to do if job is finished? only remove from actionmask?
    sample = env.siminfo.pointersample
    if env.problem.setupstatejob[machine][setup][job]["mean"] == 0
        if sample !== nothing
            env.siminfo.nextevent[machine][1] = (env.siminfo.time
                                                    + env.samples[sample]["processing_time"][job][operation][machine])
        else
            env.siminfo.nextevent[machine][1] = env.siminfo.time + processsample(env.problem, job, operation, machine, rng)
        end
        
        env.siminfo.activitytimes[machine] =    [
                                                    [env.siminfo.time, env.problem.orders[job].processingtimes[operation][machine]["min"], # store min time
                                                    env.siminfo.time + env.problem.orders[job].processingtimes[operation][machine]["min"]], # store threshold for udpating
                                                    [0, env.deterministic ? env.problem.orders[job].processingtimes[operation][machine]["mean"] : mean(env.problem.orders[job].processingtimes[operation][machine]["dist"])], # store mean of stochastic part
                                                    [env.siminfo.time, env.siminfo.nextevent[machine][1]], # store real time for faster comparison later
                                                    env.deterministic ? ["None", "Determinisitic"] : ["None", env.problem.orders[job].processingtimes[operation][machine]["type"]],
                                                    [0, env.problem.orders[job].processingtimes[operation][machine]["max"]]
                                                ]
    else
        # schedule and setup
        if sample !== nothing
            tmpS = (env.siminfo.time + env.samples[sample]["setup_time"][machine][setup][job])
            env.siminfo.nextevent[machine][1] = (tmpS + env.samples[sample]["processing_time"][job][operation][machine])
        else
            tmpS = (env.siminfo.time + setupsample(env.problem,job, machine, setup,rng))
            env.siminfo.nextevent[machine][1] = (tmpS + processsample(env.problem, job, operation, machine,rng))
        end
        env.siminfo.activitytimes[machine] =    [
                                                    [env.siminfo.time + env.problem.setupstatejob[machine][setup][job]["min"],
                                                        env.problem.orders[job].processingtimes[operation][machine]["min"], # store min times
                                                        tmpS + env.problem.orders[job].processingtimes[operation][machine]["min"]], # threshold for min
                                                    [env.problem.setupstatejob[machine][setup][job]["mean"] == 0 ? 0 : ( env.deterministic ? env.problem.setupstatejob[machine][setup][job]["mean"] : mean(env.problem.setupstatejob[machine][setup][job]["dist"])),
                                                        env.deterministic ? env.problem.orders[job].processingtimes[operation][machine]["mean"] :  mean(env.problem.orders[job].processingtimes[operation][machine]["dist"])], # store mean of stochastic part
                                                    [tmpS, env.siminfo.nextevent[machine][1]], # store real time for faster comparison later
                                                    env.deterministic ? ["Determinisitic", "Determinisitic"] : [env.problem.setupstatejob[machine][setup][job]["type"], env.problem.orders[job].processingtimes[operation][machine]["type"]],
                                                    [env.problem.setupstatejob[machine][setup][job]["max"],
                                                        env.problem.orders[job].processingtimes[operation][machine]["max"]]
                                                ]


        env.siminfo.numsetups += 1
        for j ∈ env.siminfo.notfinished
            # updated based on changed setup!
            if env.siminfo.pointercurrentop[j] < length(env.problem.orders[j].ops)
                if machine ∈ env.problem.orders[j].eligmachines[env.problem.orders[j].ops[env.siminfo.pointercurrentop[j]]]
                    env.state.generalfeatures["ON"][j] += (env.problem.setupstatejob[machine][env.problem.orders[job].type][job]["mean"]
                                                        - env.problem.setupstatejob[machine][setup][job]["mean"])
                end
            end
        end
    end

    env.state.starttime[machine] = env.siminfo.time
    expmin = env.siminfo.activitytimes[machine][1][1] + env.siminfo.activitytimes[machine][1][2]
    env.state.expfinish[machine] = [expmin, env.siminfo.activitytimes[machine][2][1] + expmin + env.siminfo.activitytimes[machine][2][2]]

    if env.siminfo.flowtime[job][2] === nothing
        env.siminfo.flowtime[job][2] = env.siminfo.time
    end

    # update features
    mset = deepcopy(env.problem.orders[job].eligmachines[operation])
    denom = length(mset)
    env.state.generalfeatures["EW"][machine] += env.problem.orders[job].processingtimes[operation][machine]["mean"]*(denom-1)/denom
    env.state.generalfeatures["CW"][machine] += env.problem.orders[job].processingtimes[operation][machine]["mean"]*(denom-1)/denom
    env.state.generalfeatures["JWM"][machine] += 1 - 1/denom
    env.state.generalfeatures["CJW"][machine] += 1 - 1/denom

    deleteat!(mset, findfirst(x-> x == machine,mset))
    for m ∈ mset
        env.state.generalfeatures["EW"][m] -= env.problem.orders[job].processingtimes[operation][m]["mean"]/denom
        env.state.generalfeatures["CW"][m] -= env.problem.orders[job].processingtimes[operation][m]["mean"]/denom
        env.state.generalfeatures["JWM"][m] -= 1/denom
        env.state.generalfeatures["CJW"][m] -= 1/denom
    end

    # changes in old machines!
    if "dds" ∈ env.actionmasks
        changesS = []
        changesW = []
        dd = env.problem.orders[job].duedate
        # check changes in old machines
        orderChange = env.problem.orders[job]
        for m ∈ orderChange.eligmachines[operation]
            DDS = Inf
            DDW = Inf
            for jobs_old ∈ 1:length(env.problem.orders)
                if jobs_old != job
                    ord_old = env.problem.orders[jobs_old]
                    if env.siminfo.pointercurrentop[jobs_old] != 0
                        if m ∈ ord_old.eligmachines[env.siminfo.pointercurrentop[jobs_old]]
                            if orderChange.type == ord_old.type
                                if env.state.executable[jobs_old]
                                    if ord_old.duedate < DDS
                                        DDS = ord_old.duedate
                                    end
                                else
                                    if ord_old.duedate < DDW
                                        DDW = ord_old.duedate
                                    end
                                end
                            end
                        end
                    end
                end
            end
            if DDS > dd
                append!(changesS, m)
                env.siminfo.edd_type_machine[orderChange.type][m]["schedule"] = DDS
            end 
            if DDW > dd
                append!(changesW, m)
                env.siminfo.edd_type_machine[orderChange.type][m]["waiting"] = DDW
            end
        end
    end

    # increment operation pointer
    if env.siminfo.pointercurrentop[job] < length(env.problem.orders[job].ops)
        env.siminfo.pointercurrentop[job] += 1
        env.state.generalfeatures["RF"][job] = sum(env.problem.orders[job].eligmachines[env.siminfo.pointercurrentop[job]])
        if env.siminfo.pointercurrentop[job] < length(env.problem.orders[job].ops)
            j = env.problem.orders[job]
            nop = env.siminfo.pointercurrentop[job]
            env.state.generalfeatures["ON"][job] = sum(j.processingtimes[j.ops[nop]][m]["mean"] 
                                                    + env.problem.setupstatejob[m][env.state.setup[m]][job]["mean"]
                                                    for m ∈ j.eligmachines[j.ops[nop]]) / length(j.eligmachines[j.ops[nop]])
        else
            env.state.generalfeatures["ON"][job] = 0
        end
    else
        env.siminfo.pointercurrentop[job] = 0 # JOB is done!
        deleteat!(env.siminfo.notfinished, findfirst(x-> x == job,env.siminfo.notfinished)) 
        env.state.generalfeatures["RF"][job] = 0
        env.state.generalfeatures["ON"][job] = 0
    end
    
    # changes for new machines only waiting -> not in AIM
    if "dds" ∈ env.actionmasks && env.actionspace != "AIM"
        new_ops = env.siminfo.pointercurrentop[job]
        if new_ops != 0
            dd = env.problem.orders[job].duedate
            orderChange = env.problem.orders[job]
            for m ∈ orderChange.eligmachines[new_ops]
                DDW = Inf
                for jobs_new ∈ 1:length(env.problem.orders)
                    if jobs_new != job
                        ord_new = env.problem.orders[jobs_new]
                        if env.siminfo.pointercurrentop[jobs_new] != 0
                            if m ∈ ord_new.eligmachines[env.siminfo.pointercurrentop[jobs_new]]
                                if orderChange.type == ord_new.type
                                    if !env.state.executable[jobs_new]
                                        if ord_new.duedate < DDW
                                            DDW = ord_new.duedate
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                if DDW < dd
                    append!(changesW, m)
                    env.siminfo.edd_type_machine[orderChange.type][m]["waiting"] = dd
                end
            end
        end
    end
    updatestate!(env, action)
    updatedactionmask!(env, action)
    if "dds" ∈ env.actionmasks && (!isempty(changesS) || !isempty(changesW)) 
        updateactionmask_edd_change(env, action, changesS, changesW)
    end
end

function updateenv!(env::Env) 

    # no actions can be taken anymore -> move sim to end and calculate values!
    if isempty(env.siminfo.notfinished)
        while(true)
            tmpEvent = findmin(flagmin,arrayzip(env.siminfo.nextevent,env.siminfo.time))
            env.siminfo.nextevent[tmpEvent[2]][2] = 1
            if tmpEvent[1] == Inf
                # println("Done")
                return true
            end
            updatemetrics!(env,tmpEvent)
        end
    else
        decisionneeded = false
        while(!decisionneeded)# find next event
            tmpNext = findmin(flagmin,arrayzip(env.siminfo.nextevent,env.siminfo.time)) #tmpNext format is (time, index)
            if tmpNext[1] == Inf
                for (i,e) ∈ enumerate(env.siminfo.nextevent)
                    if env.state.occupation[i] !== nothing && e[1] > env.siminfo.time
                        env.siminfo.nextevent[i][2] = 0
                    end
                end
                if all(env.siminfo.nextevent[i][2] == 1 for i ∈ eachindex(env.siminfo.nextevent))
                    println(env.state.actionmask)
                    println(env.siminfo.notfinished)
                    state_of_unfinished_jobs = [env.siminfo.pointercurrentop[j] for j ∈ env.siminfo.notfinished]
                    println("unfinished jobs: ", state_of_unfinished_jobs)
                    occupation = findall(x -> x !== nothing , env.state.occupation)
                    ErrorMachine = occupation[1]
                    ErrorEvent = env.siminfo.nextevent[ErrorMachine]
                    error("no change! ", env.siminfo.nextevent, env.siminfo.time, env.state.occupation, env.siminfo.notfinished, ErrorEvent)
                end

            else
                if env.siminfo.time >= tmpNext[1] # no move forward
                #println("other event at same time")
                #println(env.siminfo.nextevent)
                # env.siminfo.nextevent[tmpNext[2]][2] = 1 #TODO needed?
                # no update ∈ siminfo
                # no update to state? -> #TODO  done ∈ action realization already?
                # TODO update features!!

                else # time moves forward
                    # println(env.siminfo.nextevent)
                    env.siminfo.nextevent = [[i[1],0] for i ∈ env.siminfo.nextevent] # remove flags ∈ nextevent
                    # TODO updated siminfo / nothing left to update?
                    # update reward metrics
                    # TODO update all the following features incrementally ->
                    for i ∈ keys(env.problem.orders)
                        env.state.generalfeatures["JS"][i] = max(env.state.generalfeatures["DD"][i] - env.state.generalfeatures["RW"][i] - tmpNext[1], 0)
                    end
                    env.state.generalfeatures["CT"] = tmpNext[1]
                    # println((findall(x->x[1] == tmpNext[1],env.siminfo.nextevent)))

                    all_events = findall(x->x[1] == tmpNext[1],env.siminfo.nextevent)
                    if tmpNext[1] > 0 && length(all_events) > 1
                        for index ∈ all_events
                            event = (env.siminfo.nextevent[index][1], Int(index))
                            updatemetrics!(env,event)
                            updatestate!(env,event)
                        end
                    else
                        updatemetrics!(env,tmpNext)
                        updatestate!(env,tmpNext)
                    end
                end
                
                if sum(env.state.actionmask) > 0
                    if sum(values(env.state.executable)) > 0 || macdecne(env)
                        decisionneeded = true
                        stochasticupdating!(env) # TODO no always used! onyl if wanted!!!!
                    else
                        env.siminfo.nextevent[tmpNext[2]][2] = 1
                    end
                else
                    #println("skip event machine ",tmpNext[2])
                    env.siminfo.nextevent[tmpNext[2]][2] = 1
                end
            end

        end
        return false
    end
end

"""
    function step!(env::Env, action::Tuple{Int,Int,String},rng::AbstractRNG; incrementalreward = true)

    perform on step of the simulation with
    
    inputs:
        env    -> being the environment ∈ which the step is taken
        action -> Tuple of form (jobindex, machineindex, string for action type)
        rng    -> random number generator for stochasticity // reproducability
        reward -> type of reward to be used

    returns:
        env.state -> returning the state for agent
        increment -> returning the incremental metric increase (incremental reward)
        done      -> boolean to indicate end of sim
        infofield -> tbd
"""
function step!(env::Env, action::Tuple{Int,Int,String},rng::AbstractRNG; incrementalreward = true) #TODO add type of reward here!!
    @assert (env.state.actionmask[action[1],action[2]] == 1) "action is not allowed"

    # println(env.siminfo.nextevent, ", time: " , env.siminfo.time, ", action: ", action, )
    # state_of_unfinished_jobs = [env.siminfo.pointercurrentop[j] for j ∈ env.siminfo.notfinished]
    # println("unfinished jobs: ", state_of_unfinished_jobs)
    # println(env.state.occupation)


    # println("action: ", action, ", translated machine: ", env.siminfo.mappingmachines[action[2]] )
    # metrics befreo step
    tmpMetric = [env.siminfo.time,  sum(env.siminfo.flowtime[j][1] for j ∈ eachindex(env.siminfo.flowtime)),
                                    sum(env.siminfo.tardiness[j] for j ∈ eachindex(env.siminfo.tardiness)),
                                    sum(env.siminfo.discretetardiness[j] for j ∈ eachindex(env.siminfo.discretetardiness)),
                                    env.siminfo.numtardyjobs, env.siminfo.numsetups]
    
    # [ deepcopy(env.siminfo.time), deepcopy(env.siminfo.flowtime), deepcopy(env.siminfo.tardiness),
    #        deepcopy(env.siminfo.discretetardiness), deepcopy(env.siminfo.numtardyjobs), deepcopy(env.siminfo.numsetups)]
    if action[3] == "W"
        waitingaction!(env,action)
    elseif action[3] == "R"
        risksetupaction!(env,action,rng)
    elseif action[3] == "S"
        scheduleaction!(env,action,rng)
    end
    # step to next event ∈ list and update state(mask, features, etc) and rest of siminfo (metrics,pointer, etc)
    done = updateenv!(env)
    
    if incrementalreward
        increment = -([env.siminfo.time,  sum(env.siminfo.flowtime[j][1] for j ∈ eachindex(env.siminfo.flowtime)),
                                        sum(env.siminfo.tardiness[j] for j ∈ eachindex(env.siminfo.tardiness)),
                                        sum(env.siminfo.discretetardiness[j] for j ∈ eachindex(env.siminfo.discretetardiness)),
                                        env.siminfo.numtardyjobs, env.siminfo.numsetups] - tmpMetric)
    else
        if done
            increment = -[env.siminfo.time,  sum(env.siminfo.flowtime[j][1] for j ∈ eachindex(env.siminfo.flowtime)),
                        sum(env.siminfo.tardiness[j] for j ∈ eachindex(env.siminfo.tardiness)),
                        sum(env.siminfo.discretetardiness[j] for j ∈ eachindex(env.siminfo.discretetardiness)),
                        env.siminfo.numtardyjobs, env.siminfo.numsetups]
        else
            increment = [0 for _ ∈ 1:6]
        end
    end

    # TODO return infos (take a look at libraries)
    infofield = nothing

    return env.state, increment, done, infofield
end

##########################################################################
# Helper to update state
##########################################################################
"""
update state based on an action selected, depending if it was a schedule action or setup action
"""
function updatestate!(env::Env, action::Tuple{Int,Int,String})
    job = action[1]
    machine = action[2]

    # define if schedule or only risk setup
    # if schedule
    if action[3] == "S"
        env.state.occupation[machine] = job
        env.state.setup[machine] = env.problem.orders[job].type
        env.state.executable[job] = false
    else
        # if only setup
        env.state.occupation[machine] = 0
        env.state.setup[machine] = env.problem.orders[job].type
    end
    # TODO update features
end

"""
update state based on an event happening
"""
function updatestate!(env::Env, event::Tuple{Float64,Int})
    # define job that finished
    machine = event[2]
    job = env.state.occupation[machine]
    if job != 0
        operation = env.siminfo.pointercurrentop[job] == 0 ? 
                env.problem.orders[job].ops[length(env.problem.orders[job].ops)] : 
                env.problem.orders[job].ops[env.siminfo.pointercurrentop[job] - 1]

        env.state.generalfeatures["RW"][job] -= env.problem.orders[job].orderavgwork[operation]
        env.state.generalfeatures["RO"][job] -= 1

        env.state.generalfeatures["EW"][machine] -= env.problem.orders[job].processingtimes[operation][machine]["mean"] #finished job reduces work
        env.state.generalfeatures["CW"][machine] -= env.problem.orders[job].processingtimes[operation][machine]["mean"] #finished job reduces work
        env.state.generalfeatures["JWM"][machine] -= 1 #finished job reduces work
        env.state.generalfeatures["CJW"][machine] -= 1 #finished job reduces work


        mset = deepcopy(env.problem.orders[job].eligmachines[operation])
        denom = length(mset)
        env.state.generalfeatures["CW"][machine] += env.problem.orders[job].processingtimes[operation][machine]["mean"]*(denom-1)/denom
        env.state.generalfeatures["CJW"][machine] += 1/denom #
    end

    # remove machine occupation
    env.state.occupation[machine] = nothing
    # set job to executable
    if job ∈ env.siminfo.notfinished 
        env.state.executable[job] = true
    end
    if "dds" ∈ env.actionmasks update_edd!(env) end
    updatedactionmask!(env)
    # TODO update features
end
##########################################################################

function stochasticupdating!(env::Env)
    for (m,i) ∈ filter(n -> n[2] !== nothing, env.state.occupation)

        if i == 0 # only setup 
            if env.siminfo.time > env.siminfo.activitytimes[m][1][1] # > threshold for min
                # stochastic update for setup
                if env.siminfo.activitytimes[m][4][1] == "Exponential"
                    env.state.expfinish[m][2] = env.siminfo.time + env.siminfo.activitytimes[m][2][1] # add only stochastic part to current time
                elseif env.siminfo.activitytimes[m][4][1] == "Uniform"
                    env.state.expfinish[m][2] = (env.siminfo.time + env.siminfo.activitytimes[m][1][1] + env.siminfo.activitytimes[m][5][1])/2
                end
            end

            #TODO update more features e.g. workload! slack... ???

        else # scheduled setup and processing
            if env.siminfo.activitytimes[m][1][1] < env.siminfo.time < env.siminfo.activitytimes[m][3][1] # between min and real time 
                # stochastic update for setup
                if env.siminfo.activitytimes[m][4][1] == "Exponential"
                    env.state.expfinish[m][2] = env.siminfo.time + env.siminfo.activitytimes[m][2][1] + #stochastic part of setup
                                            env.siminfo.activitytimes[m][1][2] + env.siminfo.activitytimes[m][2][2] # min + stochastic part of processing
                elseif env.siminfo.activitytimes[m][4][1] == "Uniform"
                    env.state.expfinish[m][2] = (env.siminfo.time + env.siminfo.activitytimes[m][1][1] + env.siminfo.activitytimes[m][5][1])/2 + #stochastic part of setup
                                            env.siminfo.activitytimes[m][1][2] + env.siminfo.activitytimes[m][2][2] # min + stochastic part of processing
                end
                
                env.state.expfinish[m][1] = env.siminfo.time + env.siminfo.activitytimes[m][1][2]
                #TODO update more features e.g. workload! slack... ???

            elseif env.siminfo.activitytimes[m][3][1] <= env.siminfo.time < env.siminfo.activitytimes[m][1][3] #setup is completed. not stochastic anymore!
                env.state.expfinish[m][2] = env.siminfo.activitytimes[m][1][3] + env.siminfo.activitytimes[m][2][2]
                env.state.expfinish[m][1] = env.siminfo.activitytimes[m][1][3]
                #TODO update more features e.g. workload! slack... ???

            elseif env.siminfo.activitytimes[m][1][3] <= env.siminfo.time  #processing min part is done.
                if env.siminfo.activitytimes[m][4][2] == "Exponential"
                    env.state.expfinish[m][2] = env.siminfo.time + env.siminfo.activitytimes[m][2][2]
                elseif env.siminfo.activitytimes[m][4][2] == "Uniform"
                    env.state.expfinish[m][2] = (env.siminfo.time + env.siminfo.activitytimes[m][1][3] + env.siminfo.activitytimes[m][5][2])/2
                end
                #TODO update more features e.g. workload! slack... ???
            end

        end
    end
end

# Helper to update values ∈ siminfo
function updatemetrics!(env::Env,event)
    env.siminfo.time = event[1] 
    jobid = env.state.occupation[event[2]]
    if jobid !==nothing && jobid != 0
        incrementflow!(env, event, jobid)
        incrementtard!(env, event, jobid)
        incrementdistard!(env, event, jobid)
    end
end

function incrementflow!(env::Env,event,jobID)
    @assert (env.siminfo.flowtime[jobID][2] !== nothing) "can not finish a job that never started / arrived"
    env.siminfo.flowtime[jobID][1] = event[1] - env.siminfo.flowtime[jobID][2]
end

function incrementtard!(env::Env,event,jobID)
    tmpD = env.problem.orders[jobID].duedate
    if event[1] > tmpD
        if env.siminfo.tardiness[jobID] == 0
            env.siminfo.numtardyjobs += 1
        end
        env.siminfo.tardiness[jobID] = event[1] - tmpD
    end
end

function incrementdistard!(env::Env,event,jobID)
    # TODO if defined discrete duedate structure
end
