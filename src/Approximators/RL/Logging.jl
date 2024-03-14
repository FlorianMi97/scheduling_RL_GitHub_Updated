function logging(a::Union{Agent{E2E},Agent{WPF}},i,r,al,cl,el,l,n,ev) 
    append!(a.logger["reward"], r)

    if i == 1
        a.logger["actor_loss"] = [al]
        a.logger["critic_loss"] = [cl]
        a.logger["entropy_loss"] = [el]
        a.logger["loss"] = [l]
        a.logger["exp_var"] = [ev]
        a.logger["rewardlastdone"] = [n]

    else
        append!(a.logger["actor_loss"], al)
        append!(a.logger["critic_loss"], cl)
        append!(a.logger["entropy_loss"], el)
        append!(a.logger["loss"], l)
        append!(a.logger["exp_var"], ev)
        append!(a.logger["rewardlastdone"], n)
    end
end

function TBCallback(logger, agent::Union{Agent{E2E},Agent{WPF}}, al ,cl, el, l, n, r, wr, ev)
    
    # model = agent.approximator.policy.AC
    # param_dict = Dict{String, Any}()
    # fill_param_dict!(param_dict, model, "")
    with_logger(logger) do
        @info "losses" actor_loss = al critic_loss = cl entropy_loss = el loss = l
        @info "norm" norm = n exp_var = ev
        # @info "model" params=param_dict log_step_increment=0
        @info  "training" avg_reward = r best_reward = maximum(wr) worst_reward = minimum(wr)
    end
end

function TBCallbackEval(logger, agent::Union{Agent{E2E},Agent{WPF}}, avgreward, gap)
    with_logger(logger) do
        @info  "eval" avg_objective = avgreward gap = gap
    end
end


function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end