mutable struct E2E <: AbstractApproximator
    policy::PPOPolicy
    envs::Vector{Env}
    objective::Vector{Any} # TODO use this for reward not for objective
    trainable::Bool
end

# constructor
function createE2E(ac::ActorCritic, env, obj; kwargs...)
    E2E(PPOPolicy(ac, env[1], actionsize(env[1]); kwargs...), env, obj,true) # TODO multiple envs...
end

function train!(a::Agent{E2E}, generations, evalevery, testenvs, finaleval; showinfo = false, showprogressbar = true, TBlog = false)
    if evalevery > 0 && !TBlog
        @warn "Eval is not stored since no TBlogger is active"
    end
    policy = a.approximator.policy
    for env in a.approximator.policy.workers
        resetenv!(env) # reset envs before starting training
    end

    if showprogressbar p = Progress(generations, 1, "Training E2E") end
    if TBlog logger = TBLogger("logs/$(a.problemInstance)/$(a.type)_$(a.actionspace)", tb_increment) end #TODO change overwrite via kwargs? but log files are large!

    for i in 1:generations
        update!(policy, a.objective)
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

        # logging(a,i,r,al,cl,el,l,n,ev) not needed anymore?

        if evalevery > 0 && TBlog
            if i % evalevery == 0
                a.model = policy.AC
                values = testagent(a, testenvs)
                TBCallbackEval(logger, a, values[1], values[6])
                #TODO logging of value
            end
        end

    end
    a.model = policy.AC

    if finaleval
        values = testagent(a, testenvs)
        if TBlog TBCallbackEval(logger, a, values[1], values[6]) end
        return values
    end
end

function test(a::Agent{E2E},env,nrseeds)
    testE2E(a.model.actor, env, a.objective, nrseeds, a.rng)
end

function nextaction(a::Agent{E2E},env)
    state = flatstate(env.state)
    mask = vec(!=(0).(env.state.actionmask)) # transform to vector of Bools if it exists
    actionindex = getaction(a.model.actor, state, mask, is_sampling = false)
    translateaction(actionindex, env, nothing)
end

function testE2E(actor, env, objective, nrsamples, rng)
    metrics = []
    fitness = []
    gaps = []
    objective = -objective

    for i in 1:nrsamples
        isempty(env.samples) ? pointer = nothing : pointer = i
        tmpfitness, tmpmetrics = evalE2E(actor, env, objective, pointer, rng)
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

function evalE2E(actor, env, objective, pointer, rng)
    metrics = [0,0,0,0,0,0]
    # reset env
    resetenv!(env)
    setsamplepointer!(env.siminfo, pointer)
    t = false
    state = flatstate(env.state)
    while !t
        mask = vec(!=(0).(env.state.actionmask)) # transform to vector of Bools if it exists
        actionindex = getaction(actor, state, mask, is_sampling = false)
        action = translateaction(actionindex, env, nothing)
        nextstate, rewards, t, info = step!(env, action, rng)
        metrics += rewards
    end

    fitness = dot(objective,metrics)
    return fitness, metrics
end