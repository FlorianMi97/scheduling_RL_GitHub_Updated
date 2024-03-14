"""
    used to store memory for masked actionspaces! otherwise remove one CircularArrayBuffer
"""
struct Trajectory{D}
    state::Vector{CircularArrayBuffer}
    action::Vector{CircularArrayBuffer} # TODO  only for discrete actions? continous needs dict with values since multiple needed?
    reward::Vector{CircularArrayBuffer}
    value::Vector{CircularArrayBuffer}
    terminal::Vector{CircularArrayBuffer}
    action_log_prob::Vector{CircularArrayBuffer}
    advantage::Vector{CircularArrayBuffer}
    returns::Vector{CircularArrayBuffer}
    mask::Union{Vector{CircularArrayBuffer}, Nothing}

end

function Trajectory(D,n_workers,max_time_steps,statesize, actionsize, masked)
    s = [CircularArrayBuffer{Float32}(statesize,max_time_steps) for _ in 1:n_workers]

    if D == Categorical
        a = [CircularArrayBuffer{Int}(max_time_steps) for _ in 1:n_workers]
        alp = [CircularArrayBuffer{Float32}(max_time_steps) for _ in 1:n_workers]
    else
        a = [CircularArrayBuffer{Float32}(actionsize, max_time_steps) for _ in 1:n_workers]
        alp = [CircularArrayBuffer{Float32}(actionsize, max_time_steps) for _ in 1:n_workers]
    end
    
    m = ifelse(masked, [CircularArrayBuffer{Bool}(actionsize, max_time_steps) for _ in 1:n_workers], nothing)
    r = [CircularArrayBuffer{Float32}(max_time_steps) for _ in 1:n_workers]
    v = [CircularArrayBuffer{Float32}(max_time_steps) for _ in 1:n_workers]
    d = [CircularArrayBuffer{Bool}(max_time_steps) for _ in 1:n_workers]
    adv = [CircularArrayBuffer{Float32}(max_time_steps) for _ in 1:n_workers]
    ret = [CircularArrayBuffer{Float32}(max_time_steps) for _ in 1:n_workers]
    Trajectory{D}(s,a,r,v,d,alp,adv,ret,m)
end


function record!(t::Trajectory,worker,s,a,r,d,alp,m)
    push!(t.state[worker],s)
    push!(t.action[worker],a)
    push!(t.reward[worker],r)
    push!(t.terminal[worker],d)
    push!(t.action_log_prob[worker],alp)
    if m !== nothing
        push!(t.mask[worker],m)
    end
end

function record!(t::Trajectory,worker,v::Vector)
    for value in v
        push!(t.value[worker],value) # push!(t.value[worker],v...)
    end
end

function store!(t::Trajectory,worker,adv,ret)
    append!(t.advantage[worker],adv)
    append!(t.returns[worker],ret)
end

function sample_exp(t::Trajectory,inds::Vector{Tuple{Int,Int}})
    s = Array{Float32,2}(undef,size(t.state[1],1),length(inds))
    a = ifelse(t isa Trajectory{Categorical}, [], Array{Float32,2}(undef,size(t.action[1],1),length(inds)))
    mask = t.mask !== nothing ? Array{Bool,2}(undef,size(t.mask[1],1),length(inds)) : nothing

    v = []
    log_p = []
    ret = []
    adv = []
    for (i,ind) in enumerate(inds)
        worker = ind[1]
        time = ind[2]
        s[:,i] = t.state[worker][:, time]
        if t.mask !== nothing
            mask[:,i] = t.mask[worker][:, time]
        end
        if t isa Trajectory{Categorical}
            push!(a,t.action[worker][time])
        else
            a[:,i] = t.action[worker][:, time]
        end
        push!(v,t.value[worker][time])
        push!(log_p,t.action_log_prob[worker][time])
        push!(ret,t.returns[worker][time])
        push!(adv,t.advantage[worker][time])
    end
    return s, a, v, log_p, ret, adv, mask
end

function vectorize(cab::Vector{CircularArrayBuffer})
    tmp = []

    for i in 1:length(cab)
        tmp = vcat(tmp, cab[i])
    end
    tmp
end