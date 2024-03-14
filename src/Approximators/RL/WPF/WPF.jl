struct Priorityfunction
    expression
    nn_mask::Vector{Bool}
    max::Vector{Float32}
    min::Vector{Float32}
end

function Priorityfunction(s::String,
                max::Union{Vector{Number},Number}=1,
                min::Union{Vector{Number},Number}=0)

    features = ["PT","DD","RO","RW","ON","JS","RF","CT","EW","JWM","ST","NI","NW","SLA","CW","CJW","TT"]
    mask = [true for _ in features]

    for i in eachindex(features)
        s = replace(s, features[i] => "w.ω_" * features[i] * " * f." * features[i])
        mask[i] = occursin(features[i],s)
    end

    f = Meta.parse("(w,f) -> " * s)
    expr = eval(f)

    if max isa Number
        max = [max for i in mask if i]
    end
    if min isa Number
        min = [min for i in mask if i]
    end
    if sum(mask) == 0
        error("no features selected")
    end
    if max isa Vector{Number}
        if length(max) != sum(mask)
            error("max vector has wrong length")
        end
    end
    if min isa Vector{Number}
        if length(min) != sum(mask)
            error("min vector has wrong length")
        end
    end

    Priorityfunction(expr,mask,max,min)
end

function PriorityfunctionfromTree(tree)
    # TODO generate an expression from a tree input!
    error("not done yet")
end


mutable struct WPF <: AbstractApproximator
    policy::PPOPolicy{ActorCritic{GaussianNetwork},Normal}
    envs::Vector{Env}
    objective::Vector{Any}
    priorityfunction::Priorityfunction
    trainable::Bool 
end

function createWPF(ac::ActorCritic, env, obj, prio; kwargs...)
    WPF(PPOPolicy(ac, env[1],  numberweights(prio); dist = Normal, kwargs...), env, obj, prio,true) # TODO multiple envs...
end

function train!(a::Agent{WPF}, generations, evalevery, testenvs, finaleval; showinfo = false, showprogressbar = true, TBlog = false)
    if evalevery > 0 && !TBlog
        @warn "Eval is not stored since no TBlogger is active"
    end
    
    policy = a.approximator.policy
    for env in a.approximator.policy.workers # reset envs before starting training
        resetenv!(env)
    end

    if showprogressbar p = Progress(generations, 1, "Training WPF") end
    if TBlog logger = TBLogger("logs/$(a.problemInstance)/$(a.type)_$(a.actionspace)", tb_increment) end #TODO change overwrite via kwargs? but log files are large!
    for i in 1:generations
        update!(policy, a.objective, a.approximator.priorityfunction)
         
        r = mean([x for x in policy.rewardlastdone if x != 0])
        if isempty(r)
            r = 0
        end
        al = mean(policy.actor_loss)
        cl = mean(policy.critic_loss)
        el = mean(policy.entropy_loss)
        l = mean(policy.loss)
        n = mean(policy.norm)
        ev = policy.exp_var

        if TBlog TBCallback(logger, a ,al, cl, el, l, n, r,[x for x in policy.rewardlastdone if x != 0], ev) end

        if showprogressbar
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
        if showinfo
            println("actor loss after iteration $i: ", al)
            println("critic loss after iteration $i: ", cl)
            println("entropy loss after iteration $i: ", el)
            println("loss after iteration $i: ", l)
            println("norm after iteration $i: ", n)
            println("mean reward after iteration $i: ", r)
            println("explained variance after iteration $i: ", ev)
        end

        # logging(a,i,r,al,cl,el,l,n,ev)

        if evalevery > 0 && TBlog
            if i % evalevery == 0
                a.model = [policy.AC, a.approximator.priorityfunction]
                values = testagent(a, testenvs)
                TBCallbackEval(logger, a, values[1], values[6])
                #TODO logging of value
            end
        end


    end
    a.model = [policy.AC, a.approximator.priorityfunction]

    if finaleval
        values = testagent(a, testenvs)
        if TBlog TBCallbackEval(logger, a, values[1], values[6]) end
        return values
    end
end

function test(a::Agent{WPF},env,nrseeds)
    testWPF(a.model[1].actor, a.model[2], env, a.objective, nrseeds, a.rng)
end

function nextaction(a::Agent{WPF},env)
    state = flatstate(env.state)
    prio = a.approximator.priorityfunction
    mask = [prio.nn_mask; prio.max; prio.min] 
    actionindex = getaction(a.model[1].actor, state, mask, is_sampling = false)[1] # TODO use σ ?
    translateaction(actionindex, env, a.model[2])
end

function testWPF(actor,prio, env, objective, nrsamples,rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    mask = [prio.nn_mask; prio.max; prio.min] #TODO vary between different envs?

    for i in 1:nrsamples
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = evalWPF(actor, mask, prio, env, objective, pointer, rng)
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

    return sum(fitness),fitness, gaps
end

function evalWPF(actor, mask, prio, env, objective, pointer, rng)
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo,pointer)
    t = false
    state = flatstate(env.state)
    while !t
        #TODO define action
 
        actionindex = getaction(actor, state, mask, is_sampling = false)[1] # TODO use σ ?
        action = translateaction(actionindex, env, prio)
        nextstate, rewards, t, info = step!(env, action, rng)

        metrics += rewards
    end

    fitness = dot(objective,metrics)
    return fitness, metrics
end