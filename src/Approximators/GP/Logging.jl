
logging(a::Agent{GP}, ngen, lastbest, improvement, uniquetrees, avgage) = begin
    append!(a.logger["reward"], lastbest)

    if ngen == 0
        a.logger["improvement"] = [0]
        a.logger["unique_trees"] = [uniquetrees]
        a.logger["average_age"] = [avgage]
    else
        append!(a.logger["improvement"], improvement)
        append!(a.logger["unique_trees"], uniquetrees)
        append!(a.logger["average_age"], avgage)
    end
end

function TBCallback(logger, population, lastbest, improvement, uniquetrees, avgage)
    densitymax = maximum(x -> x[5] ,population)
    densitybest = population[1][5]
    sizebest = sizeofrule(population[1][1][1])                              
    with_logger(logger) do
        
        @info "training" avg_reward = -lastbest # TODO add best and worst as well? from samples best_reward = maximum(wr) worst_reward = minimum(wr)
        @info "metrics" number_uniquetrees = uniquetrees average_age = avgage improvement = improvement density_best = densitybest density_max = densitymax size_best = sizebest
        
        #TODO add more!!!!
        # e.g. show best tree as infix?
        # population mean fitness
        # ...
    end
end

function TBCallbackEval(logger, avgreward, gap)
    with_logger(logger) do
        @info  "eval" avg_objective = avgreward gap = gap
    end
end