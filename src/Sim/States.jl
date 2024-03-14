abstract type AbstractState end

Base.@kwdef struct InitialState <: AbstractState
    setupstate
    occupation
    starttime
    expfinish #[min,exp]
    generalfeatures::Dict
    priofeatures::Dict
    executable
    normalize
    actionmask
end
function InitialState(problem::Problem, ad::String, am::Array{String})

    tmpA = defineactionmask(problem,am)

    InitialState(   setupstate = Dict{Int, Union{Nothing, Int}}(m => nothing for m in problem.machines),
                    occupation = Dict{Int, Union{Nothing, Int}}(m => nothing for m in problem.machines),
                    starttime = Dict(m=>0.0 for m in problem.machines),
                    expfinish = Dict(m=>[0.0, 0.0] for m in problem.machines),
                    generalfeatures = problem.generalfeatures,
                    priofeatures = problem.priorityfeatures,
                    executable = Dict(j=>true for j in keys(problem.orders)),
                    normalize = problem.normalize_features,
                    actionmask = tmpA)
end
Base.@kwdef mutable struct State <: AbstractState
    setup
    occupation
    starttime
    expfinish
    generalfeatures::Dict
    priofeatures::Dict
    executable
    normalize # format [norm factor, min]
    actionmask
end

function State(init::InitialState)
    State(  setup = deepcopy(init.setupstate),
            occupation = deepcopy(init.occupation),
            starttime = deepcopy(init.starttime),
            expfinish = deepcopy(init.expfinish),
            generalfeatures = deepcopy(init.generalfeatures),
            priofeatures = deepcopy(init.priofeatures),
            executable = deepcopy(init.executable),
            normalize = deepcopy(init.normalize),
            actionmask = deepcopy(init.actionmask)
        )
end

function statesize(env::AbstractEnv)
    length(flatstate(env.state))
end

function actionsize(env::AbstractEnv)
    t = size(env.state.actionmask)
    return t[1]*t[2]
end

function flatstate(s::State, normalize::Bool=true)
    vecstate = Vector{Float32}()

    if normalize
        append!(vecstate,scalestateRL.(
            translatecategoricaldata.([s.occupation[k] for k in sort(collect(keys(s.occupation)))]),
            s.normalize["occupation"][2],
            s.normalize["occupation"][1]
            ))
        append!(vecstate, scalestateRL.(
            translatecategoricaldata.([s.setup[k] for k in sort(collect(keys(s.setup)))]),
            s.normalize["setupstate"][2],
            s.normalize["setupstate"][1]
            ))
        append!(vecstate,scalestateRL.(
            translatecategoricaldata.([s.starttime[k] for k in sort(collect(keys(s.starttime)))]),
            s.normalize["CT"][2],
            s.normalize["CT"][1]
            ))
        append!(vecstate,scalestateRL.(
            translatecategoricaldata.([s.expfinish[k][1] for k in sort(collect(keys(s.expfinish)))]),
            s.normalize["CT"][2],
            s.normalize["CT"][1]
            ))
        append!(vecstate,scalestateRL.(
            translatecategoricaldata.([s.expfinish[k][2] for k in sort(collect(keys(s.expfinish)))]),
            s.normalize["CT"][2],
            s.normalize["CT"][1]
            ))
        
        # TODO check why the follwoign is not working? should be faster?
        # tmp = Vector{Float32}()
        # append!(tmp,translatecategoricaldata.([s.starttime[k] for k in sort(collect(keys(s.starttime)))]))
        # append!(tmp,translatecategoricaldata.([s.expfinish[k][1] for k in sort(collect(keys(s.expfinish)))]))
        # append!(tmp,translatecategoricaldata.([s.expfinish[k][2] for k in sort(collect(keys(s.expfinish)))]))
        
        # append!(vecstate, scalestateRL.(tmp,
        #     s.normalize["CT"][2],
        #     s.normalize["CT"][1]
        #     ))

        append!(vecstate,scalestateRL.(
            translatecategoricaldata.([s.executable[k] for k in sort(collect(keys(s.executable)))]),
            s.normalize["executable"][2],
            s.normalize["executable"][1]
            ))

        for (k,v) in s.generalfeatures
            append!(vecstate,scalestateRL.(translatecategoricaldata.([v[kk] for kk in sort(collect(keys(v)))]),
            s.normalize[k][2],
            s.normalize[k][1]))
        end
    else
        append!(vecstate,translatecategoricaldata.([s.occupation[k] for k in sort(collect(keys(s.occupation)))]))
        append!(vecstate,translatecategoricaldata.([s.setup[k] for k in sort(collect(keys(s.setup)))]))
        append!(vecstate,translatecategoricaldata.([s.starttime[k] for k in sort(collect(keys(s.starttime)))]))
        append!(vecstate,translatecategoricaldata.([s.expfinish[k][1] for k in sort(collect(keys(s.expfinish)))]))
        append!(vecstate,translatecategoricaldata.([s.expfinish[k][2] for k in sort(collect(keys(s.expfinish)))]))
        append!(vecstate,translatecategoricaldata.([s.executable[k] for k in sort(collect(keys(s.executable)))]))
        for v in values(s.generalfeatures)
            append!(vecstate,translatecategoricaldata.([v[k] for k in sort(collect(keys(v)))]))
        end
    end
    vecstate
end

function translatecategoricaldata(d)
    if d ===  nothing #TODO better encoding of nothing?
        return Float32(-1.0)
    elseif d == true
        return Float32(1.0)
    elseif d == false 
        return Float32(0.0)
    else
        return Float32(d)
    end
end

# scale in [-1,1]
scalestateRL(x,min,max) = begin
    if min == max
        return Float32(0.0)
    end
    Float32((x-min) / (max-min) *2.0 -1.0)
end
# scale in [0,1]
scalefeatureGP(x,min,max) = begin
    if min == max
        return 0.0
    end
    (x-min) / (max-min)
end