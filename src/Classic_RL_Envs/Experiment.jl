using Flux
using ParameterSchedulers
using ProgressMeter
using Functors: @functor
using ChainRulesCore
using CircularArrayBuffers
using Distributions
using Random
using StatsBase

using TensorBoardLogger, Logging

include("Cartpole.jl")
include("../Approximators/RL/Networks.jl")
include("../Approximators/RL/Trajectory.jl")
include("../Approximators/RL//PPO.jl")

include("../Approximators/RL/Utils.jl")

function train!(policy, generations, logger)

    for env in policy.workers
        resetenv!(env) # reset envs before starting training
    end

    p = Progress(generations, 1, "cartpole")
    for i in 1:generations
        update!(policy)

        r = mean([x for x in policy.rewardlastdone if x != 0])
        al = mean(policy.actor_loss)
        cl = mean(policy.critic_loss)
        el = mean(policy.entropy_loss)
        l = mean(policy.loss)
        n = mean(policy.norm)
        ev = policy.exp_var

        TBCallback(logger, policy.AC,al ,cl, el, l, n, r, ev)



        ProgressMeter.next!(p; showvalues = [
                                            (:actor_loss, al),
                                            (:critic_loss, cl),
                                            (:entropy_loss, el),
                                            (:loss, l),
                                            (:norm, n),
                                            (:mean_reward, r),
                                            (:explained_variance, ev)
                                            ]
                            )
    end
end

function update!(p::PPOPolicy)
    t = p.trajectory
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
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range

    vf_clip = p.vf_clip

    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight

    #---------------------------------------------------------------------------------------------------------------
    # ROLLOUT PHASE: create experience from epochs!
    #---------------------------------------------------------------------------------------------------------------
    for (i,w) in enumerate(workers) #TODO parallelised distribut via CUDA.jl
        state = w.state
        nextstate = nothing
        for _ in 1:n_steps

            mask = nothing

            actoraction, logprob = getaction(AC.actor, state, mask, p.current_trajectory, p.n_random_start, rng)

            action = actoraction
            nextstate, reward, done, info = step!(w, action)
            p.accreward[i] += reward # accumulate reward

            record!(t, i, state, actoraction, reward, done, logprob, mask)
            if done
                resetenv!(w) # reset workers env only if it is done! -> otherwise not able to always reach end!
                p.rewardlastdone[i] = p.accreward[i] # save reward of last done trajectory
                p.accreward[i] = 0.0 # reset accumulated reward
                @assert p.rewardlastdone[i] != 0
            end
            state = w.state
        end
        values = AC.critic(t.state[i]) |> vec #done here for better vectorizing and the option to parallelize rollout worker with a gpu?
        record!(t, i, values)
        # calculate advantage and returns
        advantages = generalized_advantage_estimation(AC,t.reward[i],t.value[i],γ,λ,t.terminal[i],nextstate)
        returns = advantages .+ t.value[i]
        store!(t, i, advantages, returns)
    end

    # println(t)

    # println(t.action_log_prob)
    
    #---------------------------------------------------------------------------------------------------------------
    # LEARNING PHASE: update based on experience!
    #---------------------------------------------------------------------------------------------------------------
    
    # update learning rate and entropy loss if scheduler is given
    useschedulers!(p) #TODO add scheduler for clip_range?
    
    # generate microbatches for sgd
    n_exp = n_steps * length(workers)
    indicestuple = collect(Iterators.product(1:length(workers),1:n_steps))
    @assert n_exp % n_microbatches == 0 "batch size mismatch"
    microbatch_size = n_exp ÷ n_microbatches
    for epoch in 1:n_epochs # use recorded data, i.e. trajectories n:epoch times
        rand_indices = shuffle!(rng, indicestuple) # shuffle trajectories for sgd
        for i in 1:n_microbatches # performe sgd for batch
            inds = rand_indices[(i-1)*microbatch_size+1: i*microbatch_size]
            
            s, a, v, log_p, r, adv, mask = sample_exp(t,inds) # sample from trajectories

            # println("s : ", s)
            
            # normalize advantages over micro batch via:
            ifelse(norm_adv, (adv = adv .- mean(adv)) ./ (std(adv) .+ 1e-8), adv)

            ps = Flux.params(AC)
            gs = gradient(ps) do
                v′ = AC.critic(s) |> vec # calculate new values

                # println("v′ : ", v′)
                
                if AC.actor isa GaussianNetwork 
                    x = AC.actor.pre(s)
                    μ, σ = AC.actor.μ(x), AC.actor.σ(x)
                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims=1))
                    else
                        log_p′ₐ = normlogpdf(μ, σ, a)
                    end
                    entropy_loss = mean(size(σ, 1) * (log(2.0f0π) + 1) .+ sum(log.(σ .+ 1.0f-10 ); dims=1)) / 2
                
                else
                    # actor is assumed to return discrete logits
                    raw_logit′ = AC.actor(s) 
                     
                    logit′ = raw_logit′
                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)

                    # log_p′ₐ = log_p′[a]
                    dist = [Categorical(x; check_args=false) for x in eachcol(p′)]
                    log_p′ₐ = logpdf.(dist, a)

                    entropy_loss = -sum(infmul.(p′, log_p′)) * 1 // size(p′, 2)

                end

                ratio = exp.(log_p′ₐ .- log_p)
                surr1 = ratio .* adv 
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv
                actor_loss = -mean(min.(surr1, surr2))

                if vf_clip !== nothing
                    unclipped = (r .- v′) .^ 2
                    clipped = v .+ clamp.(v′ .- v, -vf_clip, vf_clip) 
                    clipped = (r .- clipped) .^ 2
                    critic_loss = mean(max.(unclipped, clipped))
                else
                    critic_loss = mean((r .- v′) .^ 2)
                end
                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                if isnan(entropy_loss)
                    println("entropy_loss: ", entropy_loss)
                    println("entropy los is: ", typeof(entropy_loss))
                end

                @assert !any(isnan.(actor_loss)) "actor_loss is NaN"
                @assert !any(isnan.(critic_loss)) "critic_loss is NaN"

                ignore_derivatives() do
                    p.actor_loss[i, epoch] = actor_loss
                    p.critic_loss[i, epoch] = critic_loss
                    p.entropy_loss[i, epoch] = entropy_loss
                    p.loss[i, epoch] = loss 
                end

                loss
            end

            p.norm[i, epoch] = clip_by_global_norm!(gs, ps, p.max_grad_norm) # this clips the gradient gs by L2 norm before it is used to update the parameters ps

            if !all(isfinite, p.norm[i,epoch])
                error("norm is not finite")
            end

            Flux.Optimise.update!(AC.optimizer, Flux.params(AC), gs) 
            
        end
    end
    ypred = vectorize(t.value)
    ytrue = vectorize(t.returns)
    vartrue = StatsBase.var(ytrue)
    p.exp_var = vartrue != 0.0 ? (1 - StatsBase.var(ytrue - ypred) / vartrue) : NaN
    p.current_trajectory +=1
end

function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

function TBCallback(logger, model, al ,cl, el, l, n, r, ev)
                                            
    param_dict = Dict{String, Any}()
    fill_param_dict!(param_dict, model, "")
    with_logger(logger) do
        @info "actor_loss" actor_loss = al
        @info "critic_loss" critic_loss = cl
        @info "entropy_loss" entropy_loss = el
        @info "loss" loss = l
        @info "norm" norm = n

        @info "model" params=param_dict log_step_increment=0
        @info  "avg_reward" avg_reward = r
        @info "exp_var" exp_var = ev
    end
end


env = CartPoleEnv()
ac = ActorCritic(
    actor = Chain(Dense(4 => 64),Dense(64=> 64), Dense(64 => 2)), # no specified activation function = linear activation
    critic = Chain(Dense(4 => 64),Dense(64=> 32), Dense(32 => 1)),
    optimizer = Adam(0.008))

# p = PPOPolicy(ac, env, 2, masked = false, n_steps = 10, n_workers = 1, n_microbatches = 1, clip_range=0.12f0,)

p = PPOPolicy(ac, env, 2, masked = false, n_steps = 1000, clip_range=0.12f0,)

logger = TBLogger("logs/cartpole",tb_overwrite)

train!(p, 500, logger)