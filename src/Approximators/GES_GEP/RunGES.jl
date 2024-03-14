function train!(a::Agent{GES}, generations, evalevery, testenvs, finaleval; showinfo = false, showprogressbar = true, TBlog = false)
    if evalevery > 0 && !TBlog
        @warn "Eval is not stored since no TBlogger is active"
    end
    
    ges = a.approximator
    lastbest = 0 # last best value initialized
    improvement = 0
    for env in ges.envs # reset envs before starting training
        resetenv!(env)
    end

    if showprogressbar p = Progress(generations, 1, "Training Weights") end
    if TBlog logger = TBLogger("logs/$(a.problemInstance)/$(a.type)_$(a.actionspace)", tb_increment) end #TODO change overwrite via kwargs? but log files are large!

    # loop for GEP
    for ngen in 1:generations

        update!(ges)
        ges.generation += 1

        # TODO define values to log and logger function
        # if TBlog TBCallback(logger) end

        if showprogressbar
            # TODO define values to show in progressbar
            # ProgressMeter.next!(p; showvalues = [
            #                                     (:actor_loss, al),
            #                                     (:critic_loss, cl),
            #                                     (:entropy_loss, el),
            #                                     (:loss, l),
            #                                     (:norm, n),
            #                                     (:mean_reward, r),
            #                                     (:explained_variance, ev)
            #                                     ]
            #                     )
        end

        if showinfo
            println("\n", "CURRENT GENERATION: ", ngen, "\n")
            println("Weights in current generation: ", ges.bestweights)

        end

        if evalevery > 0 # && TBlog
            if ngen % evalevery == 0
                a.model = [ges.bestweights, ges.priorityfunctions] #TODO non global
                values = testagent(a, testenvs)
                if ngen == 1
                    println("Improvement not available in gen 1")
                else
                    improvement = (lastbest - values[1]) / lastbest
                    println("improvement in % after iteration $ngen: ", improvement)
                end
                
                lastbest = values[1]
                println("best value after iteration $ngen: ", lastbest)
                #TODO logging of value
                #if TBlog TBCallbackEval(logger, a, values[1], values[6]) end
                
            end
        end



    end

    a.model = [ges.bestweights, ges.priorityfunctions]

    if finaleval
        values = testagent(a, testenvs)
        if TBlog TBCallbackEval(logger, a, values[1], values[6]) end
        return values
    end
end

function update!(ges::GES)

    if ges.optimizer == "GradientDecent"
        gradientdecent!(ges)
        
    elseif ges.optimizer == "ADAM"
        #todo
    else
        error("Optimizer not supported")
    end

end

function gradientdecent!(ges::GES)

    priority_functions = ges.priorityfunctions
    μ = vcat(ges.bestweights...)
    σ = vcat(ges.sigma...)
    #println("Initial μ (weights): ", μ)
    #println("Initial σ: ", σ)

    N = Int(ges.popSize/2) # Number of perturbations to evaluate
    D = length(μ) # Number of weights
    α_μ = ges.learningrates[1]  # Define learning rate for μ
    α_σ = ges.learningrates[2]  # Define learning rate for σ

    # Initialize Δμ and Δσ
    Δμ = zeros(D)
    Δσ = zeros(D)

    # Perturbation and evaluation loop
    weights_pos = []
    weights_neg = []

    ε_values = []
    for n=1:N  # Loop over the number of pertubation (!= samples)
        weights_p =[]
        weights_n = []
        ε_val = []
        for i = 1:D
            # Draw perturbation ε ~ N(0, σ^2)
            ε = randn() * σ[i]  
            w⁺ = μ[i] .+ ε
            w⁻ = μ[i] .- ε
            append!(weights_p, w⁺)  
            append!(weights_n, w⁻)
            append!(ε_val, ε)
            #println("Perturbation ε[", i, "]: ", ε, " | w⁺: ", w⁺, " | w⁻: ", w⁻) 
        end
        push!(weights_pos, weights_p)
        push!(weights_neg, weights_n)
        push!(ε_values, ε_val)


                
    end    
    #helper function to segment weights again
    function segment_weights(flat_weights, ges::GES)
        segmented_weights = []
        offset = 1
        for stage_index in 1:length(ges.priorityfunctions)
            stage_length = length(ges.bestweights[stage_index])
            stage_weights = flat_weights[offset:(offset + stage_length - 1)]
            push!(segmented_weights, stage_weights)
            offset += stage_length
        end
        return segmented_weights
    end

    # Evaluate: f(w⁺) and f(w⁻) by simulation
    f_w⁺ = []
    f_w⁻ = []
    for n=1:N # Loop over the number of pertubation (!= samples)

        # Segment the weights for positive and negative perturbation
        segmented_weights_pos = segment_weights(weights_pos[n], ges)
        segmented_weights_neg = segment_weights(weights_neg[n], ges)
        #println("Segmented weights for w⁺[", n, "]: ", segmented_weights_pos)

        #fitness values for all samples_generations and over all stages 
        _, _, f_w_plus = detFitness(ges.priorityfunctions, segmented_weights_pos, ges) #TODO adapt for non global GES
        _,_, f_w_minus = detFitness(ges.priorityfunctions, segmented_weights_neg, ges) #TODO adapt for non global GES
        push!(f_w⁺, f_w_plus)
        push!(f_w⁻, f_w_minus)
        #println("Fitness for w⁺[", n, "]: ", f_w_plus, " | Fitness for w⁻[", n, "]: ", f_w_minus)
    end

    #Normalize the performance over the samples
    f_w⁺_N=[]
    f_w⁻_N=[]
    for i in eachindex(f_w⁺)
        push!(f_w⁺_N, (f_w⁺[i] - mean(f_w⁺))/std(f_w⁺))
        push!(f_w⁻_N, (f_w⁻[i] - mean(f_w⁻))/std(f_w⁻))
    end



    # Calculate gradients

    for i = 1:D
        Δμ_i = 0.0
        Δσ_i = 0.0
        for n = 1:N
            Δμ_i += (ε_values[n][i]/σ[i])  * (f_w⁺_N[n] - f_w⁻_N[n])[1]
            Δσ_i += (((ε_values[n][i])^2 - σ[i])/σ[i]) * (f_w⁺_N[n] - f_w⁻_N[n])[1]
        end

    # Update parameters μ and σ
        Δμ[i] = Δμ_i / N
        Δσ[i] = Δσ_i / N
        μ[i] += α_μ * Δμ[i]
        σ[i] += α_σ * Δσ[i]
        #println("μ[", i, "] after update: ", μ[i], " | σ[", i, "] after update: ", σ[i])
    end

    #println("Final μ (weights) in current generation: ", μ)
    #println("Final σ in current generationn: ", σ)

    # Update the GES structure 
    offset = 1
    for stage_index in 1:length(ges.priorityfunctions)
        stage_length = length(ges.bestweights[stage_index])
        ges.bestweights[stage_index] = μ[offset:(offset + stage_length - 1)]
        ges.sigma[stage_index] = σ[offset:(offset + stage_length - 1)]
        offset += stage_length
    end

    if ges.generation % ges.decayStepSize == 0
        ges.learningrates[1] *= ges.learningrates_decay[1]
        ges.learningrates[2] *= ges.learningrates_decay[2] 
    end
 
end


function adamoptimizer!(ges::GES)
    # TODO
    ges.bestweights = rand(Float32, numberweights(ges.priorityfunction))
end

function evalGESsample(priorityfunctions::Vector{Priorityfunction},weights, env, objective, pointer, rng, type="greedy")
    """
    Evaluates an individual's fitness based on a single sample or scenario
    """
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    t = false
    if type == "greedy"
    state = flatstate(env.state)
        while !t
            action = translateaction(priorityfunctions, weights, env)  #TODO adapt for non-global GES
            nextstate, rewards, t, info = step!(env, action, rng)
            metrics += rewards
        end
    else
        error("Other than 'Greedy' not implemented yet")
    end
    fitness = dot(objective,metrics)
    return fitness, metrics
end

function evalGESfitness(priorityfunctions::Vector{Priorityfunction}, weights, env, objective, nrsamples, rng, sample_tens_digit; type = "greedy")
    """
    aggregates the fitness evaluations from multiple samples to determine an individual's overall fitness
    """
    metrics = []
    fitness = []

    for i in 1:nrsamples
        
        pointer = isempty(env.samples) ? nothing : (i + nrsamples * sample_tens_digit) 
        tmpfitness, tmpmetrics = evalGESsample(priorityfunctions, weights, env, objective, pointer, rng, type)
        append!(metrics, tmpmetrics)
        append!(fitness, tmpfitness)
    end

    return sum(fitness), fitness, metrics
end

function detFitness(priorityfunctions::Vector{Priorityfunction},weights,ges::GES)
    """
    Determines the fitness of an individual potentially across multiple environments
    """
    fitness = 0
    for e in eachindex(ges.envs)
        tmpF, tmpO = evalGESfitness(priorityfunctions, weights, ges.envs[e], ges.objective, ges.samples_generation, ges.rng, ges.sample_tens_digit)
        fitness += (tmpF / ges.samples_generation)
    end
    fitness /= length(ges.envs)

    #if gep.markerdensity
    #    Ftuple = tuple([prio,weights, fitness,  get_marker(indi,gp)[1], 0]) # NOT IMPELEMENTED FOR GES 
    #else
    Ftuple = tuple(priorityfunctions, weights, fitness)
    #end
    return Ftuple
end


function test(a::Agent{GES},env,nrseeds)
    if a.approximator.granularity == "global"

        testGES(a, a.approximator.priorityfunctions ,a.approximator.bestweights, env, a.objective, nrseeds, a.rng)
    else
        error("non global granularity not implemented yet")
    end
end

function testGES(a, priorityfunctions::Vector{Priorityfunction}, weights, env, objective, nrsamples ,rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    if a.approximator.granularity == "global"
               
        for i in 1:nrsamples
            isempty(env.samples) ? pointer = nothing : pointer = i
            tmpfitness, tmpmetrics = evalGESsample(priorityfunctions, weights, env, objective, pointer, rng)
            append!(metrics, tmpmetrics)
            append!(fitness, tmpfitness)
    
            if env.type == "usecase"
                if objective == [-1.0,-0.0,-0.0,-0.0,-0.0,-0.0]
                    piobjective = env.samples[pointer]["objective_[1, 0]"]
                    append!(gaps, (tmpfitness/piobjective) -1)
                else
                    piobjective = env.samples[pointer]["objective_[0, 1]"]
                    append!(gaps, (tmpfitness - piobjective))
                end
            else
                piobjective = env.samples[pointer]["objective"]
                append!(gaps, (tmpfitness/piobjective) -1)
            end
        end
    
    else
        error("non global granularity not implemented yet")
    end

 

    return sum(fitness),fitness, gaps
end

function nextaction(a::Agent{GES},env)
    # TODO has to be adapted for non global GES
    translateaction(a.model[1], env, a.model[2])
end