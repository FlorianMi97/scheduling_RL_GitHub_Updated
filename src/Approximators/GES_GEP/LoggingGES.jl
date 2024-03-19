
logging(a::Agent{GES}, ngen, weights, lastbest, improvement, Adam_state=nothing) = begin
    append!(a.logger["reward"], lastbest)
    # Ensure "improvement" key exists
    haskey(a.logger, "improvement") || (a.logger["improvement"] = [])
    append!(a.logger["improvement"], improvement)

    haskey(a.logger, "weights") || (a.logger["weights"] = [])
    append!(a.logger["weights"], weights)

    if Adam_state !== nothing
        haskey(a.logger, "ADAM_state") || (a.logger["ADAM_state"] = [])
        append!(a.logger["ADAM_state"], Adam_state)
    end
end

function TBCallback(a::Agent{GES}, logger, weights, lastbest, improvement)
    
    

    with_logger(logger) do
        
        @info "training" avg_reward = -lastbest # TODO add best and worst as well? from samples best_reward = maximum(wr) worst_reward = minimum(wr)
        @info "metrics" improvement = improvement
        @info "weights" current_weights = weights 
        
       
    end
end

function TBCallbackEval(a::Agent{GES},logger, avgreward, gap)
    with_logger(logger) do
        @info  "eval" avg_objective = avgreward gap = gap
    end
end
