function test_rule(rule, eva_seeds, test_problems)
    results = Dict()
    for p in eachindex(test_problems)
        results[p] = Dict()
        objArray = []
        mkspArray = []
        tdArray = []
        flwtArray = []
        ntjArray = []
        nStpsArray = []
        mean = 0
        samples = sample_uncertainties(test_problems[p], length(eva_seeds), eva_seeds)
        for s in eachindex(eva_seeds)
            if s == length(eva_seeds)
                sample_results = evaluate_fitness_gantt([get_expression(rule[k],GENPRO) for k in 1:GENPRO.stages], test_problems[p], samples[s], true)
            else
                sample_results = evaluate_fitness_testing([get_expression(rule[k],GENPRO) for k in 1:GENPRO.stages], test_problems[p], samples[s])
            end
                append!(objArray, sample_results[1]); append!(mkspArray, sample_results[2]); append!(tdArray, sample_results[3]); append!(flwtArray, sample_results[4]); append!(ntjArray, sample_results[5]); append!(nStpsArray, sample_results[6]); mean += sample_results[1]
        end
        mean /= length(eva_seeds)    
        results[p]["objective"] = objArray; results[p]["makespan"] = mkspArray; results[p]["tardiness"] = tdArray; results[p]["flowtime"] = flwtArray; results[p]["ntardy"] = ntjArray; results[p]["nSetups"] = nStpsArray;
        results[p]["mean_objective"] = deepcopy(mean)
        results[p]["rule"] = Dict()
        results[p]["rule2"] = Dict()
        results[p]["number_jobs"] = length(test_problems[p].orders)
        for s in 1:GENPRO.stages
            results[p]["rule"][s] = string(get_expression(rule[s], GENPRO))
            results[p]["rule2"][s] = convert_rule(rule[s])
        end
    end
    return results
end

function get_sequences(rule, eva_seeds, test_problems)
    results = Dict()
    for p in eachindex(test_problems)
        results[p] = Dict()
        objArray = []
        mkspArray = []
        tdArray = []
        flwtArray = []
        ntjArray = []
        nStpsArray = []
        sequences = []
        samples = sample_uncertainties(test_problems[p], length(eva_seeds), eva_seeds)
        for s in 1:length(eva_seeds)
            sample_results = evaluate_fitness_gantt([get_expression(rule[k],GENPRO) for k in 1:GENPRO.stages], test_problems[p], samples[s])
            append!(objArray, sample_results[1]); append!(mkspArray, sample_results[2]); append!(tdArray, sample_results[3]); append!(flwtArray, sample_results[4]); append!(ntjArray, sample_results[5]); append!(nStpsArray, sample_results[6]); append!(sequences, sample_results[7])
        end    
        results[p]["objective"] = objArray; results[p]["makespan"] = mkspArray; results[p]["tardiness"] = tdArray; results[p]["flowtime"] = flwtArray; results[p]["ntardy"] = ntjArray; results[p]["nSetups"] = nStpsArray; results[p]["sequences"] = sequences
        results[p]["rule"] = Dict()
        results[p]["rule2"] = Dict()
        for s in 1:GENPRO.stages
            results[p]["rule"][s] = string(get_expression(rule[s], GENPRO))
            results[p]["rule2"][s] = convert_rule(rule[s])
        end
    end
    return results
end