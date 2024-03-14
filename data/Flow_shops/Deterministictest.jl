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
    return dist["mean"]
end

function sample_uncertainty(data)
    
    sample_dict = Dict{String, Any}()
    prodict = Dict(j => Dict(o => Dict() for o in data["operations_per_product"][data["orders"][j]["product"]]) 
                                         for j in keys(data["orders"]))
    setdict = Dict{String, Dict{Any, Dict{String, Float64}}}(m => Dict(pr => Dict() for pr in data["products"])
                                            for m in data["resources"])
    merge!(sample_dict, Dict("processing_time" => prodict, "setup_time" => setdict))

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
        merge!(sample_dict,Dict("seed" => "determinisitic"))
    return sample_dict 
end



function createsamples(case::String, setting::String = "base_setting")
    dir = pwd()
    foldername = string(dir,"/data/Flow_shops/Benchmark_Kurz/",setting,"/data_",case)
    # foldername = string(dir,"/data/Flow_shops/Use_Case_Cereals/",setting)
    filesinfolder = readdir(foldername)
    # filesinfolder = [f for f in filesinfolder if occursin(".json",f)]
    filesinfolder = [f for f in filesinfolder if occursin("lhmhlh-4_ii_Dete5",f)]
    
    for filename in filesinfolder
        filepath = string(foldername,"/",filename)
        data = Dict()
        open(filepath, "r") do f
            data = JSON.parse(f)
        end
        testdict = sample_uncertainty(data)
        directory = string(foldername,"/testing/", filename[1:end-5])
        mkpath(directory)
        savedir = string(directory,"/Determinstic.json")
        open(savedir, "w") do f
            JSON.print(f,testdict,4)
        end
    end
end

# TODO loop to create all test samples for all files
createsamples("ii")
