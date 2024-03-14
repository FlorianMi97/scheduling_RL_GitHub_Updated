""" store actor critic approximator in PPO with optimizer"""
struct ActorCritic{A}
    actor::A
    optimizer # TODO remove from here? to change more easily?
    critic
end

function ActorCritic(;actor, critic, optimizer=Adam())
    ActorCritic(actor, optimizer, critic)
end

@functor ActorCritic

# functor(x::ActorCritic) =
#     (actor = x.actor, critic = x.critic), y -> ActorCritic(y.actor, y.critic, x.optimizer)

"""
    GaussianNetwork(;pre=identity, μ, σ, min_σ=0f0, max_σ=Inf32)
Returns `μ` and `σ` when called.  Create a distribution to sample from using
`Normal.(μ, σ)`. `min_σ` and `max_σ` are used to clip the output from
`σ`. `pre` is a shared body before the two heads of the NN. σ should be > 0. 
You may enforce this using a `softplus` output activation. 
"""
Base.@kwdef struct GaussianNetwork
    pre = identity
    μ
    σ
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
end

GaussianNetwork(pre, μ, σ) = GaussianNetwork(pre, μ, σ, 0.0f0, Inf32)

@functor GaussianNetwork



