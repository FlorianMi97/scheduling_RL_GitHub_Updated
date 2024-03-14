# TODO make this a module?
# include GP
include("GP/Utils.jl")
include("GP/Primitive_functions.jl")
include("GP/GP.jl")
include("GP/Wrapper_GP.jl")
include("GP/Logging.jl")
include("GP/Run.jl")
include("GP/Tree.jl")

# include PPO basics
include("RL/Networks.jl")
include("RL/Trajectory.jl")
include("RL//PPO.jl")

# include E2E
include("RL/E2E/E2E.jl")
include("RL/E2E/Wrapper_E2E.jl")

# include GES
include("GES_GEP/Priofunction.jl")
include("GES_GEP/GES.jl")
include("GES_GEP/Wrapper_Prio_rule.jl")
include("GES_GEP/RunGES.jl")

# include WPF
# include("RL/WPF/WPF.jl")
# include("RL/WPF/Wrapper_WPF.jl")

# bugfix #TODO remove!
mutable struct WPF <: AbstractApproximator
    empty::Bool
end

#include Heuristics
include("Heuristics/Heuristics.jl")

include("RL/Utils.jl")
include("RL/Logging.jl")

function setobj!(a::AbstractApproximator, obj::Vector{Float64})
    if length(a.objective) >0
        for _ in eachindex(a.objective)
            deleteat!(a.objective,1)
        end
        append!(a.objective,obj)
    else
        append!(a.objective,obj)
    end
end

# TODO seperate objective and rewards? -> objective is the thing we want to optimize, rewards are the things we get from the environment



