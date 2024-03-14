mutable struct Box
    low::Array
    high::Array
    shape::Tuple
end

import StatsBase: sample

function Box(low::Number, high::Number, shape::Union{Tuple, Array{Int64, 1}}, dtype::Union{DataType, Nothing}=nothing)
    if isnothing(dtype)
        dtype = high == 255 ? UInt8 : Float32
        @warn "dtype was autodetected as $(dtype). Please provide explicit data type."
    end

    if low > high
        @warn "low  > high. Swapping values to preserve sanity"
        (low, high) = (high, low)  # Preserves sanity if low > high
    end

    if dtype <: Integer
        if !isa(low, Integer) || !isa(high, Integer)
            @warn "dtype is an Integer, but the values are floating points. Using ceiling of lower bound and floor of upper bound"
        end
        low = ceil(dtype, low)
        high = floor(dtype, high)
    end

    Low = dtype(low) .+ zeros(dtype, shape)
    High = dtype(high) .+ zeros(dtype, shape)
    return Box(Low, High, shape)
end

function Box(low::Array, high::Array, dtype::Union{DataType, Nothing}=nothing)
    @assert size(low) == size(high) "Dimension mismatch between low and high arrays."
    shape = size(low)
    @assert all(low .< high) "elements of low must be lesser than their respective counterparts in high"

    if isnothing(dtype)
        dtype = all(high .== 255) ? UInt8 : Float32
        @warn "dtype was autodetected as $(dtype). Please provide explicit data type."
    end
    if dtype <: Integer
        if !all(isa.(low, Integer)) || !all(isa(high, Integer))
            @warn "dtype is an Integer, but the values are floating points. Using ceiling of lower bound and floor of upper bound"
        end
        low = ceil.(dtype, low)
        high = floor.(dtype, high)
    else
        low = dtype.(low)
        high = dtype.(high)
    end
    return Box(low, high, shape)
end
#=
function seed!(box_obj::Box, seed::Int)
    box_obj.seed = seed
end
=#

Base.:(==)(box_obj::Box, other::Box) = checkvalidtypes(box_obj, other) && isapprox(box_obj.low, other.low) && isapprox(box_obj.high, other.high)

function sample(box_obj::Box)
    dtype = eltype(box_obj.low)
    dtype <: AbstractFloat ?
        rand(dtype, size(box_obj)) .* (box_obj.high .- box_obj.low) .+ box_obj.low :
        rand.(UnitRange.(box_obj.low, box_obj.high))
end

function contains(x::Union{Real, AbstractArray, NTuple}, box_obj::Box)
    isa(x, Number) && size(box_obj.low) == (1,) && (x = [x])
    size(x) == size(box_obj) && all(box_obj.low .<= x .<= box_obj.high)
end

function checkvalidtypes(box_obj1::Box, box_obj2::Box)
    dtype1, dtype2 = eltype(box_obj1.low), eltype(box_obj2.low)
    dtype1 == dtype2 ||                            # If the dtypes of both boxes are not the same...
            (dtype1 <: Unsigned && dtype2 <: Unsigned) || (dtype1 <: Signed && dtype2 <: Signed)  # then check if they're both signed or both unsigned.
end

mutable struct Discrete
    n::Int
    shape::Tuple
    Discrete(N::Int) = new(N, (N, ))
end

sample(discrete_obj::Discrete) = rand(1:discrete_obj.n)

function contains(x::Union{Number, AbstractArray}, discrete_obj::Discrete)
    as_int = nothing
    try
        as_int = Int.(x)
    catch InexactError
        return false
    end
    return all(1 .<= as_int .<= discrete_obj.n)
end

Base.:(==)(discrete_obj::Discrete, other::Discrete) = discrete_obj.n == other.n
Base.length(discrete_obj::Discrete) = discrete_obj.n

abstract type Env end

mutable struct CartPoleEnv <: Env
    gravity::Float32
    masscart::Float32
    masspole::Float32
    total_mass::Float32
    length::Float32  # actually half the pole's length
    polemass_length::Float32
    force_mag::Float32
    τ::Float32   # seconds between state updates
    kinematics_integrator::AbstractString

    # Angle at which to fail the episode
    θ_threshold_radians::Float32
    x_threshold::Float32
    action_space::Discrete
    observation_space::Box
    viewer
    state

    steps_beyond_done
end

function CartPoleEnv()
    gravity = 98f-1
    masscart = 1f0
    masspole = 1f-1
    total_mass = masspole + masscart
    length = 5f-1 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 1f1
    τ = 2f-2  # seconds between state updates
    kinematics_integrator = "euler"

    # Angle at which to fail the episode
    θ_threshold_radians = Float32(12 * 2 * π / 360)
    x_threshold = 24f-1

    # Angle limit set to 2θ_threshold_radians so failing observation is still within bounds
    high = [
        2x_threshold,
        maxintfloat(Float32),
        2θ_threshold_radians,
        maxintfloat(Float32)]

    action_space = Discrete(2)
    observation_space = Box(-high, high, Float32)

    viewer = nothing
    state = nothing

    steps_beyond_done = nothing
    CartPoleEnv(
        gravity, masscart, masspole, total_mass, length, polemass_length,
        force_mag, τ, kinematics_integrator, θ_threshold_radians, x_threshold,
        action_space, observation_space, viewer, state, steps_beyond_done)
end

function step!(env::CartPoleEnv, action)
    # @assert action ∈ env.action_space "$action in $(env.action_space) invalid"
    state = env.state
    x, ẋ, θ, θ̇  = state[1:1], state[2:2], state[3:3], state[4:4]
    # Action is either 1 or 2. Force is either -force_mag or +force_mag
    force = (2action.-[3]) * env.force_mag
    cosθ = cos.(θ)
    sinθ = sin.(θ)
    temp = (force .+ env.polemass_length * θ̇  .^ 2 .* sinθ) / env.total_mass
    θacc = (env.gravity*sinθ .- cosθ.*temp) ./
           (env.length * (4f0/3 .- env.masspole * cosθ .^ 2 / env.total_mass))
    xacc  = temp .- env.polemass_length * θacc .* cosθ / env.total_mass
    if env.kinematics_integrator == "euler"
        x_ = x  .+ env.τ * ẋ
        ẋ_ = ẋ  .+ env.τ * xacc
        θ_ = θ  .+ env.τ * θ̇
        θ̇_ = θ̇  .+ env.τ * θacc
    else # semi-implicit euler
        ẋ_ = ẋ  .+ env.τ * xacc
        x_ = x  .+ env.τ * ẋ_
        θ̇_ = θ̇  .+ env.τ * θacc
        θ_ = θ  .+ env.τ * θ̇_
    end

    env.state = vcat(x_, ẋ_, θ_, θ̇_)
    done =  !(all(vcat(-env.x_threshold .≤ x_ .≤ env.x_threshold,
            -env.θ_threshold_radians .≤ θ_ .≤ env.θ_threshold_radians)))

    if !done
        reward = 1f0
    elseif env.steps_beyond_done === nothing
        # Pole just fell!
        env.steps_beyond_done = 0
        reward = 1f0
    else
        if env.steps_beyond_done == 0
           # @warn "You are calling 'step!()' even though this environment has already returned done = true. You should always call 'reset()' once you receive 'done = true' -- any further steps are undefined behavior."
        end
        env.steps_beyond_done += 1
        reward = 0f0
    end

    return env.state, reward, done, Dict()
end

function resetenv!(env::CartPoleEnv)
    env.state = rand(Float32, 4) * 1f-1 .- 5f-2

    if isdefined(Main, :CuArrays)
        env.state = env.state |> gpu
    end

    env.steps_beyond_done = nothing

    return env.state
end

function statesize(env::CartPoleEnv)
    4
end

