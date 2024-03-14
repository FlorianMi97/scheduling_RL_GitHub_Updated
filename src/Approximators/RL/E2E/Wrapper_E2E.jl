function translateaction(actionindex::Int, env, nothing)
    n_jobs,n_mach = size(env.state.actionmask)
    # selected action
    sa = translate1D2D(actionindex,n_jobs)

    # if env.state.actionmask[sa[1],sa[2]] != 1
    #     println("actionindex: ",actionindex)
    #     println("action: ",sa)
    #     println("mask: ",env.state.actionmask)
    #     error("error: invalid action was selected by approximator")
    # end

    @assert env.state.actionmask[sa[1],sa[2]] == 1 "error: invalid action was selected by approximator, action was inde: $actionindex , translated $(sa). if (1,1) likely mask 0 everywhere"
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
    return (sa[1],sa[2],type)
end


function translate1D2D(actionindex::Int,j::Int)
    first = actionindex % j
    second = actionindex ÷ j +1
    if first == 0
        first = j
        second -=1
    end
    (first,second)
end