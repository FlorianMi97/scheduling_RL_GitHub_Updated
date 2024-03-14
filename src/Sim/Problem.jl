Base.@kwdef struct Order
    id::String
    type::Int
    duedate::Float64
    numops::Int
    ops::Array
    eligmachines::Dict #maps operations to machines
    processingtimes::Dict
    orderavgwork::Array
end

Base.@kwdef struct Problem
    id
    type::Char # "F" or "J" SP
    prods
    ops
    ops_per_prod
    prod_ops_routing
    op_prod_mach_comp
    orders::Dict{Int,Order}
    mapindexprods
    mapindexorder
    mapindexmachine
    mapindexops
    nStages
    machines
    machine_stage
    stage_machines
    prod_op_stage
    prod_stage_op
    setupstatejob
    generalfeatures
    priorityfeatures
    normalize_features
end

function Problem(layout, instance, file; id = 1, randomOrderbooks = false, mOrders = 55, actPro = ["CEREAL_1", "CEREAL_2", "CEREAL_3"], Seed = nothing, kwargs...)
    type = layout[1] # using "J" or "F"

    if layout == "Flow_shops"
        data = Dict()
        open(file, "r") do f
            data = JSON.parse(f)
        end
        mapindexprods = Dict(i => id for (i,id) in enumerate(sort(data["products"])))
        mapprodsindex = Dict(id => i for (i,id) in enumerate(sort(data["products"])))
        prods = sort(collect(keys(mapindexprods))) # given in Int
        
        mapindexops = Dict(i => id for (i,id) in enumerate(sort(data["operations"])))
        mapopsindex = Dict(id => i for (i,id) in enumerate(sort(data["operations"])))
        ops = sort(collect(keys(mapindexops))) # given in Int

        mapindexmachine = Dict(i => id for (i,id) in enumerate(sort(data["resources"])))
        mapmachineindex = Dict(id => i for (i,id) in enumerate(sort(data["resources"])))
        machines = sort(collect(keys(mapindexmachine)))

        ops_per_prod= Dict(p => map(x -> mapopsindex[x], data["operations_per_product"][mapindexprods[p]]) for p in prods) # given in Int
        prod_ops_routing = Dict(p => Dict(o =>
                                            map(x->mapmachineindex[x],data["operation_machine_compatibility"][mapindexprods[p]][mapindexops[o]])
                                            for o in ops_per_prod[p])
                                            for p in prods)
        nStages = data["NrStages"]

        machine_stage = Dict(mapmachineindex[k] => data["resource_stage"][k]
                            for k in sort(collect(keys(data["resource_stage"]))))

        prod_op_stage = Dict(p =>
                            Dict( mapopsindex[k] => 
                            data["product_operation_stage"][mapindexprods[p]][k] 
                            for k in sort(collect(keys(data["product_operation_stage"][mapindexprods[p]]))))
                                for p in prods
                            )
        prod_stage_op = Dict(p => Dict(v => k for (k,v) in prod_op_stage[p]) for p in prods)
        stage_machines = Dict(i=>[m for m in machines if machine_stage[m]==i] for i in 1:nStages)
        op_prod_mach_comp = Dict(p=>Dict(o=>Dict(m=> (m in prod_ops_routing[p][o]) for m in machines)
                                for o in ops_per_prod[p]) for p in prods)
        
        op_time = Dict(p => Dict(o => Dict(m => data["processing_time"][mapindexprods[p]][mapindexops[o]][mapindexmachine[m]]
                                for m in prod_ops_routing[p][o]) for o in ops_per_prod[p]) for p in prods)
        setup = data["setup_time"]

        if randomOrderbooks == true
            if Seed !== nothing
                Random.seed!(Seed)
            end
            # TODO add logic to it
            error("not yet implemented")
        else
            ordersDict = data["orders"]
            mapindexorder = Dict(i => id for (i,id) in enumerate(sort(collect(keys(ordersDict)))))
            # orders, optime, prod_ops_routing, ops_per_prod
            orders = generate_orders_from_file(ordersDict, op_time, prod_ops_routing, ops_per_prod,mapprodsindex)
        end
    #elseif
    end
    setupstatejob = Dict{Any, Dict{Any, Dict{Any, Dict{String, Any}}}}(m => 
                            Dict(s => Dict(o => dictdist(setup[mapindexmachine[m]][mapindexprods[s]][mapindexprods[orders[o].type]]) 
                            for o in keys(orders)) for s in prods) for m in machines)
    
    if "initial_setup" in keys(data)
        initial_setup = data["initial_setup"]
        for m in machines
            setupstatejob[m][nothing] = Dict(o => dictdist(initial_setup[mapindexmachine[m]][mapindexprods[orders[o].type]]) for o in keys(orders))
            setupstatejob[m][nothing][nothing] = Dict("dist" => "None","mean" => 0,"min" => 0)
        end
    else
        for m in machines
            setupstatejob[m][nothing] = Dict(o => Dict("dist" => "None","mean" => 0,"min" => 0) for o in keys(orders))
            setupstatejob[m][nothing][nothing] = Dict("dist" => "None","mean" => 0,"min" => 0)
        end
    end

    # generate features
    # features suitable for NNs ?!
    gf = Dict{String, Any}()
    gf["DD"] = Dict(i => j.duedate for (i,j) in orders)
    gf["RO"] = Dict(i => length(j.ops) for (i,j) in orders)
    gf["RW"] = Dict(i => sum(j.orderavgwork) for (i,j) in orders)
    gf["ON"] = Dict(i => length(j.ops) > 1 ? 
                        (sum(j.processingtimes[j.ops[2]][m]["mean"] for m in j.eligmachines[j.ops[2]]) / length(j.eligmachines[j.ops[2]])) : 0
                        for (i,j) in orders) # includes setup time!!!
    gf["JS"] = Dict(i => (gf["DD"][i] - gf["RW"][i]) for i in keys(orders))
    gf["RF"] = Dict(i => sum(j.eligmachines[j.ops[1]]) for (i,j) in orders)
    gf["CT"] = 0

    gf["EW"] = Dict(m => sum(j.processingtimes[o][m]["mean"]/length(j.eligmachines[o]) 
                        for j in values(orders) for o in j.ops if m in j.eligmachines[o]) for m in machines)
    
    gf["CW"] = Dict(m => machineinitialjob(m,orders) ? sum(j.processingtimes[j.ops[1]][m]["mean"]/length(j.eligmachines[j.ops[1]])
                    for j in values(orders) if m in j.eligmachines[j.ops[1]]) : 0 for m in machines)
    
    gf["JWM"] = Dict(m => sum(1/length(j.eligmachines[o]) for j in values(orders)
                            for o in j.ops if m in j.eligmachines[o]) for m in machines)
    gf["CJW"] = Dict(m => machineinitialjob(m,orders) ? sum(1/length(j.eligmachines[j.ops[1]]) for j in values(orders)
                            if m in j.eligmachines[j.ops[1]]) : 0 for m in machines)

    # addtional features needed for priority rule
    pf = Dict{String, Any}()
    pf["PT"] = Dict(i => Dict(m => j.processingtimes[j.ops[1]][m]["mean"] for m in j.eligmachines[j.ops[1]])  for (i,j) in orders)
    pf["ST"] = Dict(i => Dict(m => 0 for m in j.eligmachines[j.ops[1]])  for (i,j) in orders) # no setup at start!
    pf["NI"] = Dict(i => Dict(m => 0 for m in j.eligmachines[j.ops[1]])  for (i,j) in orders)
    pf["NW"] = Dict(i => Dict(m => 0 for m in j.eligmachines[j.ops[1]])  for (i,j) in orders)
    pf["SLA"] = Dict(i => Dict(m => length(j.eligmachines[j.ops[1]]) > 1 ? 1 : 0 for m in j.eligmachines[j.ops[1]])  for (i,j) in orders)

    # TODO add features like min time, max time, ...


    # define normalize values for state features [max,min]
    normalize_features = Dict{String,Any}("setupstate" => [prods[end],-1.0])
    normalize_features["occupation"] = [length(orders),-1.0]
    normalize_features["executable"] = [1.0,0.0]


    # define normalize value by [max,min]
    normalize_features["DD"] = [maximum(i.duedate for i in values(orders)),
                                                        minimum(i.duedate for i in values(orders))]  # update min dynamically within SIM
    normalize_features["RO"] = [maximum(length(ops_per_prod[i]) for i in prods),1.0] 
    normalize_features["RW"] = [maximum(sum(i.orderavgwork) for i in values(orders)),0.0] # TODO min = min processing instead of 0?
    normalize_features["ST"] = [maximum(setupstatejob[m][s][o]["mean"] for o in keys(orders) for s in prods for m in machines),0.0] 
    
    # used for feature next as well
    normalize_features["PT"] = [maximum(orders[i].processingtimes[o][m]["mean"] + 
                                                        setupstatejob[m][s][i]["mean"]
                                                        for i in keys(orders)
                                                            for o in orders[i].ops 
                                                                for s in vcat(prods,nothing) 
                                                                    for m in orders[i].eligmachines[o]),
                                             minimum(i.processingtimes[o][m]["mean"] for i in values(orders) 
                                                    for o in i.ops for m in i.eligmachines[o])]
                                                 
    normalize_features["JS"] = [maximum((i.duedate - sum(i.orderavgwork))  for i in values(orders)),0] # TODO negative slack is possible? or cap slack calcualtion at 0!

    # TODO suitable values for RF and ON ?
    tmp_elig = [sum(j.eligmachines[o]) for j in values(orders) for o in j.ops]
    normalize_features["RF"] = [maximum(tmp_elig),minimum(tmp_elig)]

    normalize_features["ON"] = [normalize_features["PT"][1],normalize_features["PT"][2]]

    normalize_features["NI"] = [2 * normalize_features["ST"][1],0.0] 
    normalize_features["NW"] = [2 * normalize_features["ST"][1],0.0] 

    normalize_features["CT"] =  [length(orders) * normalize_features["RW"][1] ,0.0]

    normalize_features["EW"] = [maximum(gf["EW"][m] for m in machines),minimum(gf["EW"][m] for m in machines)]
    normalize_features["CW"] = [maximum(gf["CW"][m] for m in machines),minimum(gf["CW"][m] for m in machines)] #TODO has to be updated constantly!!
    normalize_features["JWM"] = [maximum(gf["JWM"][m] for m in machines),minimum(gf["JWM"][m] for m in machines)]
    normalize_features["CJW"] = [maximum(gf["CJW"][m] for m in machines),minimum(gf["CJW"][m] for m in machines)] #TODO has to be updated constantly!!

    Problem(;   id=id, type= type, prods=prods, ops=ops, ops_per_prod=ops_per_prod, prod_ops_routing = prod_ops_routing,
                op_prod_mach_comp=op_prod_mach_comp, orders=orders, nStages=nStages, machines=machines, machine_stage=machine_stage,
                stage_machines=stage_machines, mapindexprods = mapindexprods, mapindexmachine = mapindexmachine, mapindexorder = mapindexorder, mapindexops = mapindexops,
                prod_op_stage=prod_op_stage, prod_stage_op=prod_stage_op, setupstatejob=setupstatejob, generalfeatures = gf,
                priorityfeatures = pf, normalize_features = normalize_features, kwargs...)
end

function generate_orders_from_file(orders, optime, prod_ops_routing, ops_per_prod,mapping)
    orderQueue = Dict()
    for (i,j) in enumerate(sort(collect(keys(orders))))
        t = mapping[orders[j]["product"]]
        ops = ops_per_prod[t]
        e = prod_ops_routing[t]
        p = Dict(o => Dict(m => dictdist(optime[t][o][m]) for m in e[o]) for o in ops)
        orders[j]["due_date"] === nothing ? d = 0 : d = orders[j]["due_date"]
         

        orderQueue[i] = Order(  id = j,
                                type = t,
                                duedate = d, 
                                numops = length(ops),
                                ops = ops,
                                eligmachines = e, 
                                processingtimes = p, # min, mean, denominator of rate
                                orderavgwork = [get_mean_work(e,o,p) for o in ops] 
                                )
    end
    return orderQueue
end

function sampleuncertainties(problem, nSamples, rng::AbstractRNG) 
    sample_dict = Dict(s=>Dict("processing_time" => Dict(j => Dict(o => Dict() for o in problem.ops_per_prod[problem.orders[j].type]) 
                                                    for j in keys(problem.orders))
                        , "setup_time" => Dict(m => Dict(pr => Dict() for pr in vcat(problem.prods,[nothing]))
                                                    for m in problem.machines) 
                        ) for s in 1:nSamples)
    for s in 1:nSamples
        for j in keys(problem.orders)
            job = problem.orders[j]
            # prod = j.type
            for o in job.ops
                for m in job.eligmachines[o]
                    #sample operation time 
                    if job.processingtimes[o][m]["dist"] == "None"
                        sample_dict[s]["processing_time"][j][o][m] = 0.0
                    elseif job.processingtimes[o][m]["dist"] == "Deterministic"
                        sample_dict[s]["processing_time"][j][o][m] = job.processingtimes[o][m]["mean"]
                    else
                        sample_dict[s]["processing_time"][j][o][m] = job.processingtimes[o][m]["min"] + rand(rng,job.processingtimes[o][m]["dist"])
                    end
                    #sample setup time (dependent on predecessor)
                    for pr in problem.prods #setdiff(problem.prods, [prod])
                        if problem.setupstatejob[m][pr][j]["dist"] == "None"
                            sample_dict[s]["setup_time"][m][pr][j] = 0.0
                        elseif problem.setupstatejob[m][pr][j]["dist"] == "Deterministic"
                            sample_dict[s]["setup_time"][m][pr][j] = problem.setupstatejob[m][pr][j]["mean"]
                        else
                            sample_dict[s]["setup_time"][m][pr][j] = problem.setupstatejob[m][pr][j]["min"] + rand(rng,problem.setupstatejob[m][pr][j]["dist"])
                        end
                    end 
                    sample_dict[s]["setup_time"][m][nothing][j] = 0.0   
                end
            end
        end
    end
    # samplesFile = string(@__DIR__,"/Samples/",instanceName,"_samples_stochastic_CP.json")
    # open(samplesFile, "w") do f
    #     JSON.print(f,sample_dict,4)
    # end
    return sample_dict 
end

"""
    returns a sample for setup (mainly needed in DRL)
    require problem,
    jobnumber::Int
    machine::Int
    prevsetup::String
"""
function setupsample(problem,jobNr,machine,prevsetup,rng::AbstractRNG)
    if problem.setupstatejob[machine][prevsetup][jobNr]["dist"] == "None"
        return 0.0
    elseif problem.setupstatejob[machine][prevsetup][jobNr]["dist"] == "Deterministic"
        return problem.setupstatejob[machine][prevsetup][jobNr]["mean"]
    else
        return problem.setupstatejob[machine][prevsetup][jobNr]["min"] + rand(rng, problem.setupstatejob[machine][prevsetup][jobNr]["dist"])
    end
end
"""
    returns a sample for processing (mainly needed in DRL)
    require problem,
    number of the job::Int
    current operation::Int
    machine::Int
"""
function processsample(problem,jobnumber,operation, machine,rng::AbstractRNG)
    job = problem.orders[jobnumber] 
    if job.processingtimes[operation][machine]["dist"] == "None"
        return 0.0
    elseif job.processingtimes[operation][machine]["dist"] == "Deterministic"
        return job.processingtimes[operation][machine]["mean"]
    else
        return job.processingtimes[operation][machine]["min"] + rand(rng, job.processingtimes[operation][machine]["dist"])
    end
end


function get_mean_work(route,ops,optime)
    return mean([optime[ops][m]["mean"] for m in route[ops]])
end

Base.isless(a::Order, b::Order) = a.duedate < b.duedate

function dictdist(dist::Dict)
    tmpD = Dict{String, Any}()
    
    #create distribution
    supporteddist = Dict(   "Exponential" => 1,
                            "Uniform" => 2)

                            # "Normal" => 2,
                            # "LogNormal" => 2)

    # create distribuion "object" from inputs
    type = dist["type"]
    if type == "Nonexisting"
        return Dict("dist" => "None","mean" => 0,"min" => 0, "max" => 0, "type" => "None")
    elseif type == "Deterministic"
        return Dict("dist" => "Deterministic", "mean" => dist["mean"], "max" => dist["mean"], "min" => dist["min"], "type" => "Determinisitic") #if deterministic times format is {"dist", "value"}
    elseif type ∉ keys(supporteddist) 
        error("Type is: ", type, ". Distribution is not supported. Please  don't use appreviations and check for typos")
    end

    global distributionparameters = dist["parameters"]
    if length(distributionparameters) != supporteddist[type]
        error("wrong amount of parameters for distribution")
    end
    min = dist["min"]
    if haskey(dist, "max")
        max = dist["max"]
    else
        max = nothing
    end
    typeS = Symbol(type)
    tmpD["dist"] = @eval $typeS(distributionparameters...)
    
    # truncation only possible for normal distribution
    # TODO allow truncation if mean is provided?
    if type == "Normal"
        if max !== nothing # truncated distribution
            tmpD["dist"] = truncated(d, lower = 0, upper = max-min)
        else
            tmpD["dist"] = truncated(d, lower = 0)
        end
    end

    tmpD["min"] = min
    tmpD["mean"] = mean(tmpD["dist"]) + min
    tmpD["max"] = max
    tmpD["type"] = type

    return tmpD
end

function machineinitialjob(machine,orders)
    for j in values(orders)
        if machine in j.eligmachines[j.ops[1]]
            return true
        end
    end
    return false
end
