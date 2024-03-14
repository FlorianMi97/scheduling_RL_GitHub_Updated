"assuming rewards and values are vectors -> advantages as vector

terminal vector of the same length as rewards"
function generalized_advantage_estimation(
        AC::ActorCritic,
        rewards,
        values,
        γ::T,
        λ::T,
        terminal,
        last_obs,
    ) where {T <: Number}
    advantages = similar(rewards)
    gae = 0
    for i in length(rewards):-1:1
        is_continue = !terminal[i]
        if i == length(rewards)
            next_value = AC.critic(last_obs)[1]
        else
            next_value = values[i+1]
        end
        delta = rewards[i] + γ * next_value * is_continue - values[i]
        gae = delta + γ * λ * is_continue * gae
        advantages[i] = gae
    end
    return advantages
end

#TODO 
function getaction(actor::GaussianNetwork, state, mask, current_trajectory = 0, n_random_start = 0, rng = nothing; is_sampling::Bool=true)
    x = actor.pre(state)
    μ, raw_σ = actor.μ(x), actor.σ(x)
    σ = clamp.(raw_σ, actor.min_σ, actor.max_σ)
    if is_sampling
        if current_trajectory < n_random_start 
            z = randn(rng, Float32, size(μ)) # TODO how to implement this!!!
        else
            z = ignore_derivatives() do
                noise = randn(rng, Float32, size(μ))
                clamp.(μ .+ σ .* noise,-1.0 ,1.0) # ensure action is in [-1,1]
            end
        end
        logp_π = sum(normlogpdf(μ, σ, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims=1)
        return z, logp_π
    else
        return μ, σ
    end
end

"""
    generates action from probabilities for categorical network
"""
function getaction(actor, state, mask, current_trajectory = 0, n_random_start = 0, rng = nothing; is_sampling::Bool=true)
    logits = actor(state)
    if !isnothing(mask)
        logits .+= ifelse.(mask, 0.0f0, typemin(Float32))
    end
    # logits = logits |> softmax |> send_to_host # TODO figure out how to implement send to device
    if is_sampling
        if current_trajectory < n_random_start
            dist =[
                    Categorical(fill(1 / length(x), length(x)); check_args=false) for
                    x in eachcol(logits)
                ]
        else
            logits = logits |> softmax
            dist = [Categorical(x; check_args=false) for x in eachcol(logits)]
        end
        action = rand.(rng, dist)
        action_log_prob = logpdf.(dist, action)
        return action[1], vec(action_log_prob)[1]
    else
        argmax(logits) # TODO return distrbution as well as measurement of certainty of action?
    end
end


function clip_by_global_norm!(gs, ps, clip_norm::Float32)
    gn = global_norm(gs, ps)
    if clip_norm < gn
        for p in ps
            gs[p] .*= (clip_norm / gn)
        end
    end
    gn
end

global_norm(gs, ps) = sqrt(sum(mapreduce(x -> x^2, +, gs[p]) for p in ps))

infmul(x,y) = begin
    if x == 0.0 || y == 0.0 || isinf(x) || isinf(y)
        return 0.0
    else
        return x*y
    end
end


function scale(x,min,max)
    return (x + 1)/2 * (max - min) + min
end

const log2π = log(2.0f0π)

"""
    log pdf of normal distribution for GPU differentiation
"""
function normlogpdf(μ, σ, x; ϵ=1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end

numberweights(prio) = begin
    sum(prio.nn_mask)
end

mapweights(a,b) = begin
    c = zeros(length(b))
    count = 1
    for i in eachindex(b)
        if b[i] == 1
            c[i] = a[count]
            count += 1
        end
    end
    return c
end

useschedulers!(p::PPOPolicy) = begin
    if p.lr_scheduler !== nothing
        p.AC.optimizer.eta = ParameterSchedulers.next!(p.lr_scheduler)
    end
    if p.clip_scheduler !== nothing
        p.clip_range = ParameterSchedulers.next!(p.clip_scheduler)
    end
    if p.entropy_scheduler !== nothing
        p.entropy_loss_weight = ParameterSchedulers.next!(p.entropy_scheduler)
    end
end

# TODO add hooks etc here?

normalize!(reward) = begin
    r = vcat([r[:] for r in reward]...)
    m = mean(r)
    s = std(r)
    for r in reward
        r .-= m
        if s !=0
            r ./= s
        else
            r .= 0
        end
    end
end

scaleminmax!(reward) = begin
    rv = vcat([r[:] for r in reward]...)
    rmax = maximum(rv)
    rmin = minimum(rv)
    for r in reward
        r .-= rmin
        if (rmax - rmin != 0)
            r ./= (rmax - rmin)
            #r .-= 1
        else
            r .= 0
        end
    end
end

advnormalize(adv) = begin
    adv = adv .- mean(adv)
    if std(adv) != 0
        adv = adv ./ std(adv)
    else
        adv .= 0.0f0
    end
    return adv
end

advscale(adv) = begin
    advmax = maximum(adv)
    advmin = minimum(adv)
    adv = adv .- advmin
    if advmax != advmin
        adv ./= (advmax - advmin)
        #adv .-= 1
    else
        adv .= 0.0f0
    end
    return adv
end


function infcheck(x)
    if isinf(x) || isnan(x)
        return 0.0
    else
        return x
    end
end