"""
    use to parallelism later?
"""

function vecWorker(number;env)
    vectorWorker = [] 
    for _ in 1:number
        push!(vectorWorker, deepcopy(env))
    end
    return vectorWorker
end

"""
D = Distributions -> (Normal(GaussianNetwork) or Categorical)
"""
mutable struct PPOPolicy{AC,D}
    AC::AC # TODO including optimzer and learningrates? changable via hooks?
    trajectory::Trajectory{D} 
    γ::Float32 # discount rate gamma
    λ::Float32 # TD lambda weights
    clip_range::Float32 # TODO changable via hooks?
    n_microbatches::Int # number of batches to perform sgd within each epoch
    n_epochs::Int # number of epochs to perform sgd
    n_steps::Int # number of steps to perform to record trajectory
    only_done::Bool # only record trajectory if done
    workers::Vector{Env} #rollout workers
    masked::Bool # if true, the actions are masked with the action mask
    outputnodes::Int # number of output nodes for the actor network

    lr_scheduler::Union{ParameterSchedulers.Stateful,Nothing} # schedule the learning rate
    clip_scheduler::Union{ParameterSchedulers.Stateful,Nothing} # schedule the clipping range
    entropy_scheduler::Union{ParameterSchedulers.Stateful,Nothing} # schedule the coefficient for the entropy loss

    norm_adv::String # normalize advantages
    norm_rewards::String # normalize rewards
    vf_clip::Union{Nothing,Float32} # clip value function loss
    max_grad_norm::Float32 # max gradient norm for clipping
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    rng::AbstractRNG
    n_random_start::Int # defines how long the actions taken are sampled uniformly only works with discrete actions so far
    current_trajectory::Int # TODO use for n_random start

    # for logging
    norm::Matrix{Float32}
    actor_loss::Matrix{Float32}
    critic_loss::Matrix{Float32}
    entropy_loss::Matrix{Float32}
    loss::Matrix{Float32}
    exp_var::Float32
    rewardlastdone::Vector{Float32}
    accreward::Vector{Float32}

end

function PPOPolicy(
    AC,
    env::Env,
    outputnodes;
    n_random_start=0,
    γ=0.999999f0,
    λ=0.95f0,
    clip_range=0.2f0,
    n_steps=128,
    only_done = false,
    lr_scheduler = nothing,
    clip_scheduler=nothing,
    entropy_scheduler=nothing,
    masked = true,
    norm_adv="none",
    norm_rewards="none",
    vf_clip=nothing,
    max_grad_norm=0.5f0,
    n_microbatches=4,
    n_epochs=4,
    n_workers = 25,
    actor_loss_weight=1.0f0,
    critic_loss_weight=0.5f0,
    entropy_loss_weight=1f-8,
    dist=Categorical,
    rng=Random.default_rng()
)
    if masked && dist == Normal
        throw(ArgumentError("Normal distribution is not yet supported with masked actions for weigths, set masked to false"))
    end

    if norm_adv ∉ ["none", "norm", "minmax"]
        println("norm_adv set to none, since $norm_adv is not supported")
        norm_adv = "none"
    end
    if norm_rewards ∉ ["none", "norm", "minmax"]
        println("norm_rewards set to none, since $norm_rewards is not supported")
        norm_rewards = "none"
    end

    PPOPolicy{typeof(AC),dist}(
        AC,
        Trajectory(dist,n_workers,n_steps,statesize(env), outputnodes, masked), 
        γ,
        λ,
        clip_range,
        n_microbatches,
        n_epochs,
        n_steps,
        only_done,
        vecWorker(n_workers, env = env),
        masked,
        outputnodes,
        lr_scheduler,
        clip_scheduler,
        entropy_scheduler,
        norm_adv,
        norm_rewards,
        vf_clip, 
        max_grad_norm,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        rng, #TODO use?
        n_random_start,
        0,
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        zeros(Float32, n_microbatches, n_epochs),
        0.0f0,
        zeros(Float32, n_workers),
        zeros(Float32, n_workers)
    )
end

"""
    perform one update step with multiple epochs. 
"""
function update!(p::PPOPolicy,objective, priorityfunction = nothing)
    t = p.trajectory
    only_done = p.only_done
    masked = p.masked
    # required params
    rng = p.rng
    AC = p.AC
    γ = p.γ
    λ = p.λ
    n_steps = p.n_steps
    n_epochs = p.n_epochs
    workers = p.workers
    norm_adv = p.norm_adv
    norm_rewards = p.norm_rewards
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range

    vf_clip = p.vf_clip

    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight

    #---------------------------------------------------------------------------------------------------------------
    # ROLLOUT PHASE: create experience from epochs!
    #---------------------------------------------------------------------------------------------------------------
    flush!(t) # flush old observations
    nextstate = []
    for (i,w) in enumerate(workers) #TODO parallelised distribut via CUDA.jl
        state = flatstate(w.state)
        for step in 1:n_steps # acts as max steps for rollout
            mask = ifelse(
                masked && priorityfunction === nothing, 
                vec(!=(0).(w.state.actionmask))
                , nothing
            ) # transform to vector of Bools if it exists

            actoraction, logprob = getaction(AC.actor, state, mask, p.current_trajectory, p.n_random_start, rng)
            action = translateaction(actoraction, w, priorityfunction)
            _ , metrics, done, info = step!(w, action, rng, incrementalreward = false)
            if only_done && step == n_steps && !done #TODO not sure if that is a good thing to do? rather penalize waiting?
                done = true
                reward = -10000000000
            else
                reward = dot(objective,metrics)
            end
            p.accreward[i] += reward # accumulate reward

            # TODO add penalty if no done in max steps?

            record!(t, i, state, actoraction, reward, done, logprob, mask)
            if done
                resetenv!(w) # reset workers env only if it is done! -> otherwise not able to always reach end!
                p.rewardlastdone[i] = p.accreward[i] # save reward of last done trajectory
                p.accreward[i] = 0.0 # reset accumulated reward
                if only_done
                    state = flatstate(w.state)
                    break
                end
            end
            state = flatstate(w.state)
        end
        
        push!(nextstate, state)
        values = t.state[i] |> get_matrix |> AC.critic |> vec #done here for better vectorizing and the option to parallelize rollout worker with a gpu?
        record!(t, i, values)
        # # calculate advantage and returns
        # advantages = generalized_advantage_estimation(AC,t.reward[i],t.value[i],γ,λ,t.terminal[i],nextstate[i])
        # returns = advantages .+ t.value[i]
        # # returns = normalize(returns) # does this make sense here? only single trajecotry not whole batch!
        # store!(t, i, advantages, returns)
    end

    
    # normalize rewards based on trajectory batch!
    if norm_rewards == "norm"
        normalize!(t.reward)
    elseif norm_rewards == "minmax"
        scaleminmax!(t.reward)
    end
    
    # do it with a matrix not a loop???
    for i in 1:length(workers)
        # calculate advantage and returns
        advantages = generalized_advantage_estimation(AC,t.reward[i],t.value[i],γ,λ,t.terminal[i],nextstate[i])
        returns = advantages .+ t.value[i]
        # returns = normalize(returns) # does this make sense here? only single trajecotry not whole batch!
        store!(t, i, advantages, returns)
    end
    #---------------------------------------------------------------------------------------------------------------
    # LEARNING PHASE: update based on experience!
    #---------------------------------------------------------------------------------------------------------------
    
    # update learning rate and entropy loss if scheduler is given
    useschedulers!(p) #TODO add scheduler for clip_range?
    
    # generate microbatches for sgd
    n_exp = sum(map(x -> length(x),t.advantage)) # total number of observations
    indicestuple = [(i,j) for i in 1:length(t.advantage) for j in 1:length(t.advantage[i])]
    microbatch_size = ifelse(n_exp % n_microbatches == 0, n_exp ÷ n_microbatches, (n_exp ÷ n_microbatches)+ 1)

    for epoch in 1:n_epochs # use recorded data, i.e. trajectories n:epoch times
        rand_indices = shuffle!(rng, indicestuple) # shuffle trajectories for sgd
        for i in 1:n_microbatches # performe sgd for batch
            if i == n_microbatches
                inds = rand_indices[(i-1)*microbatch_size+1:end]
            else
                inds = rand_indices[(i-1)*microbatch_size+1: i*microbatch_size]
            end
            s, a, v, log_p, r, adv, mask = sample_exp(t,inds) # sample from trajectories

            if !all(isfinite, s) # check if nan
                println( "nan in state")
            end
            
            # normalize advantages over micro batch via else minmaxnorm them
            if norm_adv == "norm"
                adv = advnormalize(adv)
            elseif norm_adv == "minmax"
                adv = advscale(adv)
            end

            ps = Flux.params(AC)
            gs = gradient(ps) do
                v′ = AC.critic(s) |> vec # calculate new values
                if AC.actor isa GaussianNetwork 
                    x = AC.actor.pre(s)
                    μ, σ = AC.actor.μ(x), AC.actor.σ(x)
                    # if !all(isfinite, μ) # check if nan
                    #     println( "in $epoch , $i nan in μ")
                    # end

                    # if !all(isfinite, σ) # check if nan
                    #     println( "in $epoch , $i nan in σ")
                    # end
                    if ndims(a) == 2
                        #log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims=1))
                        log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims =1))
                    else
                        log_p′ₐ = normlogpdf(μ, σ, a)
                    end
                    entropy_loss = mean(size(σ, 1) * (log(2.0f0π) + 1) .+ sum(log.(σ .+ 1.0f-10 ); dims=1)) / 2 
                
                else
                    # actor is assumed to return discrete logits
                    raw_logit′ = AC.actor(s)
                    
                    if !all(isfinite, raw_logit′) # check if nan
                        println( "in $epoch , $i nan in raw logits")
                    end
                     
                    logit′ = ifelse(!isnothing(mask),
                         raw_logit′ .+ ifelse.(mask, 0.0f0, typemin(Float32)) 
                        ,raw_logit′)
                    p′ = softmax(logit′)

                    if !all(isfinite, p′) # check if nan
                        println( "in $epoch , $i nan in p'")
                    end
                    log_p′ = logsoftmax(logit′)
                    dist = [Categorical(x; check_args=false) for x in eachcol(p′)]
                    log_p′ₐ = logpdf.(dist, a)
                    # log_p′ₐ = log_p′[a] 
                    entropy_loss = -sum(infmul.(p′, log_p′)) * 1 // size(p′, 2)

                end
                ratio = exp.(log_p′ₐ .- log_p)
                surr1 = ratio .* adv 
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv
                actor_loss = -mean(min.(surr1, surr2))

                if vf_clip !== nothing
                    unclipped = infcheck.((r .- v′) .^ 2)
                    clipped = v .+ clamp.(v′ .- v, -vf_clip, vf_clip) 
                    clipped = infcheck.((r .- clipped) .^ 2)
                    critic_loss = mean(max.(unclipped, clipped))
                else
                    critic_loss = mean(infcheck.((r .- v′) .^ 2))
                end

                # if !all(isfinite, entropy_loss)
                #     error("in $epoch , $i entropy_loss: ", entropy_loss)
                # end

                # if !all(isfinite, actor_loss)
                #     error("in $epoch , $i actor_loss: ", actor_loss)
                # end

                # if !all(isfinite, critic_loss)
                #     error("in $epoch , $i critic_loss: ", critic_loss)
                #     println("in $epoch , $i v′: ", v′)
                # end

                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
                # println( "in $epoch , $i actor_loss: ", actor_loss)
                # println( "in $epoch , $i log_p_a: ", all(isfinite,log_p′ₐ))
                # println( "in $epoch , $i log_p: ", all(isfinite,log_p))
                # println( "in $epoch , $i ratio: ", all(isfinite, ratio))
                # println( "in $epoch , $i adv: ", all(isfinite, adv))
                # println( "in $epoch , $i surr1: ", all(isfinite,surr1))
                # println( "in $epoch , $i surr2: ", all(isfinite,surr2))
                # println( "in $epoch , $i obj actor: ", all(isfinite, min.(surr1, surr2)))
                # println( "in $epoch , $i obj actor: ", all(isfinite,mean(min.(surr1, surr2))))
                # println( "in $epoch , $i critic_loss: ", critic_loss)
                # println( "in $epoch , $i v is finite: " , all(isfinite, v))
                # println( "in $epoch , $i r is finite: " , all(isfinite, (r .- v′)))
                # println( "in $epoch , $i r is finite: " , all(isfinite, (r .- v′) .^ 2))
                # println( "in $epoch , $i r is finite: " , all(isfinite, infcheck.((r .- v′) .^ 2)))
                # println( "in $epoch , $i entropy_loss: ", entropy_loss)
                # println( "in $epoch , $i loss: ", loss)

                if !all(isfinite, loss)
                    error("in $epoch , $i loss: ", loss)
                end

                ignore_derivatives() do
                    p.actor_loss[i, epoch] = actor_loss
                    p.critic_loss[i, epoch] = critic_loss
                    p.entropy_loss[i, epoch] = entropy_loss
                    p.loss[i, epoch] = loss 
                end

                loss
            end

            p.norm[i, epoch] = clip_by_global_norm!(gs, ps, p.max_grad_norm) # this clips the gradient gs by L2 norm before it is used to update the parameters ps
            Flux.Optimise.update!(AC.optimizer, ps, gs) #TODO play with optimizer, learning rate etc.
        end
    end
    ypred = vectorize(t.value)
    ytrue = vectorize(t.returns)
    vartrue = StatsBase.var(ytrue)
    p.exp_var = vartrue != 0.0 ? (1 - StatsBase.var(ytrue - ypred) / vartrue) : NaN
    p.current_trajectory +=1
end


