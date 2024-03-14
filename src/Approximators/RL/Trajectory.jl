"""
    used to store memory for masked actionspaces! otherwise remove one CircularArrayBuffer
"""
struct Trajectory{D}
    state::Vector
    action::Vector # TODO  only for discrete actions? continous needs dict with values since multiple needed?
    reward::Vector
    value::Vector
    terminal::Vector
    action_log_prob::Vector
    advantage::Vector
    returns::Vector
    mask::Union{Vector, Nothing}

end

function Trajectory(D,n_workers,max_time_steps,statesize, actionsize, masked)
    s = [[] for _ in 1:n_workers]
    a = [[] for _ in 1:n_workers]
    alp = [[] for _ in 1:n_workers]
    m = ifelse(masked, [[] for _ in 1:n_workers], nothing)
    r = [[] for _ in 1:n_workers]
    v = [[] for _ in 1:n_workers]
    d = [[] for _ in 1:n_workers]
    adv = [[] for _ in 1:n_workers]
    ret = [[] for _ in 1:n_workers]
    Trajectory{D}(s,a,r,v,d,alp,adv,ret,m)
end


function record!(t::Trajectory,worker,s,a,r,d,alp,m)
    push!(t.state[worker],s)
    push!(t.action[worker],a)
    push!(t.reward[worker],r)
    push!(t.terminal[worker],d)
    append!(t.action_log_prob[worker],alp)
    if m !== nothing
        push!(t.mask[worker],m)
    end
end

function record!(t::Trajectory,worker,v::Vector)
    append!(t.value[worker],v)
end

function store!(t::Trajectory,worker,adv::Vector,ret::Vector)
    append!(t.advantage[worker],adv)
    append!(t.returns[worker],ret)
end

function flush!(t::Trajectory)
    for i in 1:length(t.state)
        t.state[i] = []
        t.action[i] = []
        t.reward[i] = []
        t.value[i] = []
        t.terminal[i] = []
        t.action_log_prob[i] = []
        t.advantage[i] = []
        t.returns[i] = []
        if t.mask !== nothing
            t.mask[i] = []
        end
    end
end

function sample_exp(t::Trajectory,inds::Vector{Tuple{Int,Int}})
    s = Array{Float32,2}(undef,size(t.state[1][1],1),length(inds))
    a = ifelse(isa(t,Trajectory{Categorical}), [], Array{Float32,2}(undef,size(t.action[1][1],1),length(inds)))
    mask = t.mask !== nothing ? Array{Bool,2}(undef,size(t.mask[1][1],1),length(inds)) : nothing

    v = []
    log_p = []
    ret = []
    adv = []
    for (i,ind) in enumerate(inds)
        worker = ind[1]
        time = ind[2]
        s[:,i] = t.state[worker][time]
        if t.mask !== nothing
            mask[:,i] = t.mask[worker][time]
        end
        if isa(t,Trajectory{Categorical})
            push!(a,t.action[worker][time])
        else
            a[:,i] = t.action[worker][time]
        end
        push!(v,t.value[worker][time])
        push!(log_p,t.action_log_prob[worker][time])
        push!(ret,t.returns[worker][time])
        push!(adv,t.advantage[worker][time])
    end
    return s, a, v, log_p, ret, adv, mask
end

function vectorize(cab::Vector)
    tmp = []

    for i in 1:length(cab)
        tmp = vcat(tmp, cab[i])
    end
    tmp
end

function get_matrix(cab::Vector)
    tmp = zeros(Float32, length(cab[1]), length(cab))
    for i in 1:length(cab)
        tmp[:,i] = cab[i]
    end
    tmp
end