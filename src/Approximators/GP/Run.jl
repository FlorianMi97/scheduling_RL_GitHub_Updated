function train!(a::Agent{GP}, generations, evalevery, testenvs, finaleval; showinfo = false, showprogressbar = true, TBlog = false)
    if evalevery > 0 && !TBlog
        @warn "Eval is not stored since no TBlogger is active"
    end
    
    gp = a.approximator
    lastbest = 0 # last best value initialized

    if showprogressbar p = Progress(generations, 1, "Training GP") end
    if TBlog logger = TBLogger("logs/$(a.problemInstance)/$(a.type)_$(a.actionspace)", tb_increment) end #TODO change overwrite via kwargs? but log files are large!
    # loop for GP
    # intial Pop
    pop = []    
    
    for ngen in 0:generations #TODO parallelism?
        if ngen == 0 
            pop = initialpop(gp)
        else
            newpop = next_gen!(gp, pop, ngen)   
            pop = new_pop(newpop, gp, pop, ngen)
        end
        improvement = ifelse(ngen == 0, 0 , (lastbest - pop[1][2])/lastbest)
        lastbest = pop[1][2]
        uniquetrees = length(unique(x -> x[1], pop))
        avgage = mean([x[3] for x in pop])

        if TBlog TBCallback(logger, pop, lastbest, improvement, uniquetrees, avgage) end

        if showprogressbar
            ProgressMeter.next!(p; showvalues = [
                                                (:best_value, lastbest),
                                                (:unique_trees, uniquetrees),
                                                (:avgage, avgage),
                                                (:improvement, improvement),
                                                ]
                                )
        end
        if showinfo
            println("best value after iteration $ngen: ", lastbest)
            println("AGE is : ", [pop[i][3] for i in eachindex(pop)])
            println("unique trees: ", uniquetrees)
            println("improvement in % after iteration $ngen: ", improvement)
        end

        logging(a, ngen, lastbest, improvement, uniquetrees, avgage)
        gp.generation +=1

        if evalevery > 0 && TBlog
            if ngen % evalevery == 0
                a.model = pop
                values = testagent(a, testenvs)
                TBCallbackEval(logger, values[1], values[6])
            end
        end
    end
    if !gp.stationary_samples
        pop = sortallsamples(pop,gp)
    end
    a.model = pop

    if finaleval
        values = testagent(a, testenvs)
        if TBlog TBCallbackEval(logger, values[1], values[6]) end
        return values
    end
end

function test(a::Agent{GP},env,nrseeds)
    individuum = a.model[1][1]
    testfitness([get_expression(individuum[k],a.approximator) for k in 1:a.approximator.stages], env, a.objective, nrseeds, a.rng)[1:3] # TODO use other returns? store env objective mroe detailed -> outlier!
end

function nextaction(a::Agent{GP},env)
    individuum = a.model[1][1]
    actionsfromindividuum([get_expression(individuum[k],a.approximator) for k in 1:a.approximator.stages],env)[1]
end
