using JSON
using Random
using Distributions




function compatible(data,job,machine)
    product = data["orders"][job]["product"]
    stage = data["resource_stage"][machine]
    op = [k for (k,v) in data["product_operation_stage"][product] if v == stage]
    if length(op) == 0
        return false
    end
    if machine ∈ data["operation_machine_compatibility"][product][op[1]]
        return true
    else
        return false
    end
end

function calctime(dist::Dict)

    global distributionparameters = dist["parameters"]
    typeS = Symbol(dist["type"])

    tmpDist = @eval $typeS(distributionparameters...)

    return dist["min"] + rand(tmpDist)
end

function sample_uncertainty(data, seed)
    
    sample_dict = Dict{String, Any}()
    prodict = Dict(j => Dict(o => Dict() for o in data["operations_per_product"][data["orders"][j]["product"]]) 
                                         for j in keys(data["orders"]))
    setdict = Dict{String, Dict{Any, Dict{String, Float64}}}(m => Dict(pr => Dict() for pr in data["products"])
                                            for m in data["resources"])
    merge!(sample_dict, Dict("processing_time" => prodict, "setup_time" => setdict))

    Random.seed!(seed)
        for j in keys(data["orders"])
            # job = problem.orders[j]
            prod = data["orders"][j]["product"]
            for o in data["operations_per_product"][prod]
                for m in data["operation_machine_compatibility"][prod][o]
                    
                    #sample operation time 
                    if data["processing_time"][prod][o][m]["type"] == "Nonexisting"
                        sample_dict["processing_time"][j][o][m] = 0.0
                    # elseif job.processingtimes[o][m]["dist"] == "Deterministic"
                    #     sample_dict[s]["processingtime"][j][o][m] = job.processingtimes[o][m]["mean"]
                    # else
                    # TODO no inclusion of provided max or determinsitic solutions
                    else
                        sample_dict["processing_time"][j][o][m] = calctime(data["processing_time"][prod][o][m])
                    end

                    #sample setup time (dependent on predecessor)
                    for pr in data["products"] #setdiff(problem.prods, [prod])
                        if data["setup_time"][m][pr][prod]["type"] == "Nonexisting"
                            sample_dict["setup_time"][m][pr][j] = 0.0
                        # elseif problem.setupstatejob[m][pr][j]["dist"] == "Deterministic"
                        #     sample_dict[s]["setuptime"][j][o][m][pr] = problem.setupstatejob[m][pr][j]["mean"]
                        else
                            sample_dict["setup_time"][m][pr][j] = calctime(data["setup_time"][m][pr][prod])
                        end
                    end 
                end
            end
        end
        for m in data["resources"]
            merge!(sample_dict["setup_time"][m], Dict(nothing => 
                    Dict(j => 0.0 for j in keys(data["orders"]) if compatible(data,j,m) )))
        end
        merge!(sample_dict,Dict("seed" => seed))
    return sample_dict 
end



function createsamples(case::String, setting::String = "base_setting")
    dir = pwd()
    foldername = string(dir,"/data/Flow_shops/Benchmark_Kurz/",setting,"/data_",case)
    # foldername = string(dir,"/data/Flow_shops/Use_Case_Cereals/",setting)
    filesinfolder = readdir(foldername)
    # filesinfolder = [f for f in filesinfolder if occursin(".json",f)]
    filesinfolder = [f for f in filesinfolder if occursin("lhhhha-0_ii_Expo5",f)]

    prime = [2,3,5,7,11,13,17,19,23,29,
                31,37,41,43,47,53,59,61,67,71,
                73,79,83,89,97,101,103,107,109,113,
                127,131,137,139,149,151,157,163,167,173,
                179,181,191,193,197,199,211,223,227,229,
                233,239,241,251,257,263,269,271,277,281,
                283,293,307,311,313,317,331,337,347,349,
                353,359,367,373,379,383,389,397,401,409,
                419,421,431,433,439,443,449,457,461,463,
                467,479,487,491,499,503,509,521,523,541]

    
    for filename in filesinfolder
        filepath = string(foldername,"/",filename)
        data = Dict()
        open(filepath, "r") do f
            data = JSON.parse(f)
        end
        for seed in prime
            # TODO create samples! copy from problem class
            testdict = sample_uncertainty(data, seed)
            directory = string(foldername,"/testing/", filename[1:end-5])
            mkpath(directory)
            savedir = string(directory,"/",seed, ".json")
            open(savedir, "w") do f
                JSON.print(f,testdict,4)
            end
        end
    end
end

# TODO loop to create all test samples for all files
createsamples("ii")
