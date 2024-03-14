using JSON
using Distributions


function getmean(dist::Dict)

    global distributionparameters = dist["parameters"]
    typeS = Symbol(dist["type"])

    tmpDist = @eval $typeS(distributionparameters...)

    return dist["min"] + mean(tmpDist)
end

function addmean()
    dir = pwd()

    path = "/data/Flow_shops/Use_Case_Cereals/base_setting"

    filepath = "/useCase_2_stages.json"
    filepath = dir * path * filepath
    # Load the data
    data = Dict()
    open(filepath, "r") do f
        data = JSON.parse(f)
    end

    println(keys(data))

    for (v,k) in data["setup_time"]
        for (v2,k2) in k
            for (v3,k3) in k2
                if k3["type"] != "Nonexisting"
                    mean = getmean(k3)
                    data["setup_time"][v][v2][v3]["mean"] = mean
                end  
            end
        end
    end
    for (v,k) in data["processing_time"]
        for (v2,k2) in k
            for (v3,k3) in k2
                if k3["type"] != "Nonexisting"
                    mean = getmean(k3)
                    data["processing_time"][v][v2][v3]["mean"] = mean
                end  
            end
        end
    end

    open(filepath, "w") do f
        JSON.print(f,data,4)
    end

end

addmean()