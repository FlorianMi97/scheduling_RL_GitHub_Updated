"""
    store weights for calcaulation expression via key
"""
struct Weights
    ω_PT::Float32
    ω_DD::Float32
    ω_RO::Float32
    ω_RW::Float32
    ω_ON::Float32
    ω_JS::Float32
    ω_RF::Float32
    ω_CT::Float32
    ω_EW::Float32
    ω_JWM::Float32
    ω_ST::Float32
    ω_NI::Float32
    ω_NW::Float32
    ω_SLA::Float32
    ω_CW::Float32
    ω_CJW::Float32
    ω_TT::Float32
end

# constructor via vector
Weights(v::Vector, mask::Vector) = Weights(mapweights(v,mask)...)


struct Featurevalues
    PT::Float64
    DD::Float64
    RO::Float64
    RW::Float64
    ON::Float64
    JS::Float64
    RF::Float64
    CT::Float64
    EW::Float64
    JWM::Float64
    ST::Float64
    NI::Float64
    NW::Float64
    SLA::Float64
    CW::Float64
    CJW::Float64
    TT::Float64
end

# constructor via vector
Featurevalues(env,job,machine) = begin
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
    Featurevalues(
        getPT(env,job,machine,idle,waiting),
        getDD(env,job,machine,idle,waiting),
        getRO(env,job,machine,idle,waiting),
        getRW(env,job,machine,idle,waiting),
        getON(env,job,machine,idle,waiting),
        getJS(env,job,machine,idle,waiting),
        getRF(env,job,machine,idle,waiting),
        getCT(env,job,machine,idle,waiting),
        getEW(env,job,machine,idle,waiting),
        getJWM(env,job,machine,idle,waiting),
        getST(env,job,machine,idle,waiting),
        getNI(env,job,machine,idle,waiting),
        getNW(env,job,machine,idle,waiting),
        getSLA(env,job,machine,idle,waiting),
        getCW(env,job,machine,idle,waiting),
        getCJW(env,job,machine,idle,waiting),
        getTT(env,job,machine,idle,waiting)
    )
end

"""
    generates actions based on the weights, a predefined rule, and the current state
"""
function translateaction(weights::Vector, env, priorityfunction::Priorityfunction)
    weights = scale.(weights, priorityfunction.min, priorityfunction.max)
    
    pa = [Tuple(x) for x in findall(x -> x == 1, env.state.actionmask)] # possible actions

    # generate matrix
    jobs = unique([x[1] for x in pa])
    machines = unique([x[2] for x in pa])

    priomatrix = fill(Inf,(length(jobs),length(machines)))

    # generate weights struct from vector
    w = Weights(weights,priorityfunction.nn_mask)
    

    # fill matrix
    for ix in CartesianIndices(priomatrix)
        j,m = Tuple(ix)
        if (jobs[j],machines[m]) in pa

            f = Featurevalues(env,jobs[j],machines[m])
            priomatrix[j,m] = priorityfunction.expression(w,f)
            
        end
    end

    prio,sa = findmin(priomatrix) # selected max! -> # TODO invert some features!!!
        
    # println("choosen indices are: ",sa)
    # translate pair to action tuple
    sa = (jobs[sa[1]], machines[sa[2]])

    type = ""
    # if waiting -> nothing
    if env.state.occupation[sa[2]] !== nothing
        type = "W"
    elseif env.state.executable[sa[1]] == false
        # if action design is allowing it perform risk setup
        if env.actionspace == "AIAR" #TODO
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

    # TODO add a optimized version? to the greedy one?
    return action # TODO return prio as well? 

end

