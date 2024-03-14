"""
    creates fixed structure of tree (with regard to node numbers) from maximum depth 
    returns hashmap containing the depths of individual nodes as well as
    hashmap containing the children of each node
"""
function structureTree(maxDepth::Int, maxArity::Int)
    depthNodes = Dict(0=>[1])
    children = Dict()
    parents = Dict{Int, Any}(1 => nothing)
    numNodes = sum(maxArity^i for i in 0:maxDepth)
    nodesBelow = Dict(1=>[i for i in 2:numNodes])
    for i in 1:maxDepth
        depthNodes[i] = []
        if maxArity == 2
            childrenAdd = [1,2^(maxDepth+1-i)]
        elseif maxArity == 3
            childrenAdd = [1, 1 + sum(3^j for j in 0:maxDepth-i), 3^(maxDepth+1-i)]
        else
            error("This max arity not supported")
        end
        for k in depthNodes[i-1]
            children[k] = [k+j for j in childrenAdd]
            for c in children[k]
                if i == maxDepth
                    nodesBelow[c] = []
                else
                    nodesBelow[c] = [i for i in c+1:c+sum(maxArity^j for j in 1:maxDepth-i)]
                end
                parents[c] = k
            end
            append!(depthNodes[i],children[k])
        end
    end
    for k in depthNodes[maxDepth] children[k] = [] end
    nodeDepth = Dict()
    for (v,k) in depthNodes
        for i in eachindex(k)
            nodeDepth[k[i]] = v
        end
    end

    return nodeDepth, children, nodesBelow, numNodes, parents, depthNodes 
end    

Base.@kwdef mutable struct GP <: AbstractApproximator
    envs::Vector{Env}
    stages::Int = 1
    objective::Vector{Any}
    primitives::Array{Function}
    features::Array{Symbol}
    maxDepth::Int
    primArity::Dict{Symbol,Int}
    minDepth::Int = 2
    popSize::Int = 150
    tournamentSize::Int = 5
    tournamentProb = 0.7
    childrenPerGeneration::Int = 100
    stageExchangePerGeneration::Int = 20
    oldKeep::Int = 0
    elitism::Int = 10
    probMut = 0.05
    probCro = 0.9
    randomEach::Int = 5
    
    # for pareto via marker density
    markerdensity::Bool = false
    markerstart::Int = 1
    markerend::Int = 3
    paretoTournamentSize::Int = 2

    mutationProb = 0.1
    ratioFeatureNumber = 0.75
    ratioGrow = 0.5
    ratioGrowPrimitive = 0.6
    numberUpperRange::Int = 1
    decimalStep = 0.001
    intelRule::Bool = true
    nodeDepth::Dict = Dict()
    depthNodes::Dict = Dict()
    children::Dict = Dict()
    parents::Dict = Dict()
    nodesBelow::Dict = Dict()
    numNodes::Int = 0
    primitives1::Array{Function}
    primitives2::Array{Function}
    primitives3::Array{Function}
    binarytree::Bool = true
    rng::AbstractRNG
    trainable::Bool = true

   
    pop::Array = []
    generation::Int = 0
    newpop::Array =[]

    # sampling
    stationary_samples::Bool = true # if false change samples each x generations
    change_every::Int = 10 # if stationary_samples = false, change samples every x generations
    sample_cycling::Bool = false # if true cycle through samples, otherwise set new ones
    steps_in_cycle::Int = 10 # if sample_cycling = true, x cycles before taking first again
    sample_tens_digit::Int = 0 # if sample_cycling = true, pointer to start of current sample range
    samples_generation::Int = 30 # number of samples evaluated per generation


    # TODO
    # add scheduler to change parameters

    # TODO add logging of fitness etc
    # fitness
    # number different agents
    # marker density
    # related to other added methods
    # best individual?

end

"""
    constructor of GP struct
"""
function GP(prim::Array{Function}, feat::Array{Symbol}, mD::Int, pa::Dict{Symbol, Int}, envs::Vector{Env}, objective::Vector{Any};
            rng = Random.default_rng(), kwargs...)
    nD, ch, nB, nN, p, dN = structureTree(mD, maximum(values(pa)))
    prim1 = [p for p in prim if pa[Symbol(p)] == 1]
    prim2 = [p for p in prim if pa[Symbol(p)] == 2]
    prim3 = [p for p in prim if pa[Symbol(p)] == 3]
    binarytree = isempty(prim3)

    #if binarytree
    #    error("binarytree: ", binarytree)
    #end

    GP(;primitives=prim, features=feat, maxDepth=mD, primArity = pa, envs = envs, objective = objective,
     nodeDepth=nD, depthNodes = dN, children=ch, parents = p, nodesBelow=nB, numNodes=nN, primitives1=prim1, primitives2=prim2, primitives3 = prim3, binarytree = binarytree, rng = rng, kwargs...)
end

# TODO overhaul this and add to comments
"""
    initializes the GP agent.

    parameters:         envs: Vector{Env} - vector of environments
                        objective: Vector{Any} - vector of objectives

    (optional) kwargs:  stationary_samples::Bool = true # if false change samples each x generations
                        change_every::Int = 10 # if stationary_samples = false, change samples every x generations
                        sample_cycling::Bool = false # if true cycle through samples, otherwise set new ones
                        steps_in_cycle::Int = 10 # if sample_cycling = true, x cycles before taking first again
                        samples_generation::Int = 30 # number of samples evaluated per generation

                        .....
"""
function createGP(envs; objective = [],simplerules =false, kwargs...)
    # TODO use settings to customize GP, store them somewhere to recreate agent! / save the agent.
    
                # JOB FEATURES
                # "DD"  :  job due date -> never updated use ones.
                # "RO"  :  remaining operations of job 
                # "RW"  :  remaining work of job
                # "ON"  :  average operating time of next operation of job
                # "JS"  :  job slack time (DD - RW - CT)
                # "RF"  :  routing flexibility of remaining operations of job
    
                # MACHINES FEATURES
                # "EW"  :  expected future workload of a machine (proportionally if multiple possible machines)
                # "CW"  :  current workload of a machine (proportionally if multiple possible machines)
                # "JWM" :  Future jobs waiting for machine (proportionally if multiple possible machines)
                # "CJW" :  current jobs waiting (proportionally if multiple possible machines)
    
                # JOB-MACHINES FEATURES
                # "TT"  :  total time of job machine pair (including waiting idle setup processing)
                # "PT"  :  processing time of job machine pair
                # "ST"  :  setup time of job machine pair
                # "NI"  :  required idle time of a machine
                # "NW"  :  needed waiting time of a job
                # "SLA" :  binary: 1 if setupless alternative is available when setup needs to be done, 0 otherwise
                # "BSA" :  binary: 1 if better setup alternative is available, 0 otherwise
                # "DBA" :  returns possitive difference between setup time of current setup and best alternative
    
                # GENERAL
                # "CT"  :  current time
                stringFeatures = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM","CW", "CJW", "CT","TT", "BSA", "DBA", "RF"]
                FEATURES = [Symbol(f) for f in stringFeatures]
                FEAT_STRING_DICT = Dict(FEATURES[i] => stringFeatures[i] for i in eachindex(stringFeatures))
                MAXDEPTH = 5
                arity = Dict(:add => 2, :sub => 2, :mul => 2, :div => 2, :min => 2, :max => 2, :inv => 1, :neg => 1, :squ => 1, :ifg0 => 3, :ifge0 => 3, :ifle0 => 3, :ifl0 => 3, :if0 => 3)

                PRIMITIVES = [add,sub,mul,div,min,max,neg,inv,squ,ifg0]
        
                primitivesSymbol = Dict(f => string(f) for f in PRIMITIVES)
        
                FUN_STRING_DICT = Dict(add=>"+", sub=>"-", mul=>"*", div=>"/", min=>"min", max=>"max", neg=>"neg", inv=>"inv", squ=>"squ", ifg0=>"ifg0", ifge0=>"ifge0", ifle0=>"ifle0", ifl0=>"ifl0", if0=>"if0")
                STRING_FUN_DICT = Dict(v=>k for (k,v) in FUN_STRING_DICT)

                if simplerules == true
                    arity = Dict(:add => 2, :sub => 2)
                    PRIMITIVES = [add,sub]
                end
        
                approximator = GP(PRIMITIVES, FEATURES, MAXDEPTH, arity, envs, objective; kwargs...)

end

function sample_terminal(gp::GP)
    if rand(gp.rng) <= gp.ratioFeatureNumber
        return sample(gp.rng,gp.features)
    else
        return rand(gp.rng, 0:gp.decimalStep:gp.numberUpperRange)
    end
end

"""
    initial population each depth level parameter free
    leading to a more diverse initial pop
"""
function initialpop(gp::GP)
    n = gp.samples_generation * ifelse(gp.sample_cycling, gp.steps_in_cycle, 1)
    setsamples!.(gp.envs, gp.rng, nSamples = n) #TODO number of samples specifyable, used RNG either for seeds or in uncertainity function
    population = []
    
    # of levels
    levels = gp.maxDepth - gp.minDepth +1
    tmpSize = gp.popSize ÷ levels
    
    sizeArray = []
    # create size of levels
    for _ in 1:levels
        append!(sizeArray,tmpSize)
    end
    sizeArray[end] += gp.popSize % levels 
    for (i,n) in enumerate(sizeArray)
        for x in 1:n
            individuum = []
            for _ in 1:gp.stages
                append!(individuum,[build_tree(gp, x < n/2 ? false : true; depthRegulator = gp.minDepth + i -1)])
            end
            # initialize with age = 1
            append!(population, detF(individuum,1,gp))
        end
    end
    sort!(population, by=i->i[2])
    return population
end

function build_tree(gp::GP, full=true, depth=0, number = 1, treeArray = Vector{Any}(nothing, gp.numNodes); depthRegulator = -1)
    """
    generates a random tree for the initial population with either 
    full method: tree of max depth
    or
    grow method: randomly selecting between primitives and terminals
    -------
    return a working tree array

    """

    tmpMaxDepth = depthRegulator == -1 ? gp.maxDepth : depthRegulator 

    if depth >= tmpMaxDepth
        treeArray[number] = sample_terminal(gp)    
    elseif depth < gp.minDepth || (depth < tmpMaxDepth && full) || rand(gp.rng) < gp.ratioGrowPrimitive
        usedPrim = sample(gp.rng, gp.primitives) #TODO use this!!!???
        treeArray[number] = usedPrim
        treeArray = build_tree(gp,full,depth+1,gp.children[number][1],treeArray, depthRegulator = depthRegulator)
        if gp.primArity[Symbol(usedPrim)] >= 2
            treeArray = build_tree(gp,full,depth+1,gp.children[number][2],treeArray, depthRegulator = depthRegulator)
        end
        if gp.primArity[Symbol(usedPrim)] >= 3
            treeArray = build_tree(gp,full,depth+1,gp.children[number][3],treeArray, depthRegulator = depthRegulator)
        end
    else
        treeArray[number] = sample_terminal(gp)   
    end
    return treeArray
end    

function mutation!(treeArray, gp::GP, method, minLevel=0)
    """
    mutates the object (treeArray) that calls the function by either:
    - replacing a random tree node by a new subtree including this node (method 1)
    - bitflip for a random tree node (method 2)
    - bitflip for every tree node with probability (method 3)
    - crossover with random tree "headless chicken"(method 4)

    returns the new mutated tree

    """
    applicableNodes = [i for i in 1:gp.numNodes if treeArray[i] !== nothing && gp.nodeDepth[i] >= minLevel]
    if method == 3
        for node in applicableNodes
            if rand(gp.rng) < gp.mutationProb
                element = treeArray[node]
                if element in gp.primitives
                    if gp.primArity[Symbol(element)] == 1
                        treeArray[node] = sample(gp.rng, gp.primitives1)
                    elseif gp.primArity[Symbol(element)] == 2
                        treeArray[node] = sample(gp.rng, gp.primitives2)
                    else
                        treeArray[node] = sample(gp.rng, gp.primitives3)
                    end 
                else
                    treeArray[node] = sample_terminal(gp)               
                end
            end
        end
    elseif method == 4
        tmptree = build_tree(gp, rand(gp.rng) < 0.5 ? true : false)
        treeArray = crossover_one_child(treeArray,tmptree,gp) 
    else
        mutatedNode = sample(gp.rng, applicableNodes)
        if method == 1
            for n in gp.nodesBelow[mutatedNode]
                treeArray[n] = nothing
            end
            treeArray = build_tree(gp,false,gp.nodeDepth[mutatedNode],mutatedNode,treeArray)
        else
            element = treeArray[mutatedNode]
            if element in gp.primitives
                if gp.primArity[Symbol(element)] == 1
                    treeArray[mutatedNode] = sample(gp.rng, gp.primitives1)
                elseif gp.primArity[Symbol(element)] == 2
                    treeArray[mutatedNode] = sample(gp.rng, gp.primitives2)
                else
                    treeArray[mutatedNode] = sample(gp.rng, gp.primitives3)
                end 
            else
                treeArray[mutatedNode] = sample_terminal(gp)               
            end
        end        
    end
    return treeArray
end

function crossover(parent1, parent2, gp::GP)
    crossover_points = [n for n in 2:gp.numNodes if (parent1[n] !== nothing && parent2[n] !== nothing)]
    crossover_depth = rand(gp.rng, 1:gp.maxDepth)
    crossover_options = [i for i in crossover_points if gp.nodeDepth[i] == crossover_depth]
    while crossover_options == []
        crossover_depth -= 1
        crossover_options = [i for i in crossover_points if gp.nodeDepth[i] == crossover_depth]     
    end
    crossover_cut = sample(gp.rng, crossover_options)
    child1 = Vector{Any}(nothing, gp.numNodes)
    child2 = Vector{Any}(nothing, gp.numNodes)
    for i in 1:gp.numNodes
        if i in gp.nodesBelow[crossover_cut] || i == crossover_cut
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        else
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        end
    end
    return child1, child2
end 

function crossover_one_child(parent1, parent2, gp::GP)
    crossover_points = [n for n in 2:gp.numNodes if (parent1[n] !== nothing && parent2[n] !== nothing)]
    if rand(gp.rng) < 0.9
        crossover_options = [i for i in crossover_points if (parent1[i] in gp.primitives && parent2[i] in gp.primitives)]
    else
        crossover_options = crossover_points
    end
    crossover_cut = sample(gp.rng, crossover_options)
    child1 = Vector{Any}(nothing, gp.numNodes)
    for i in 1:gp.numNodes
        if i in gp.nodesBelow[crossover_cut] || i == crossover_cut
            child1[i] = parent2[i]
        else
            child1[i] = parent1[i]
        end
    end
    return child1
end 

function crossover_guided_one_child(parent1, parent2, gp::GP)
    crossover_points = [n for n in 2:gp.numNodes if (parent1[n] !== nothing && parent2[n] !== nothing)]
    
    # select cut based on feature and subtree importance


    crossover_cut = sample(gp.rng, crossover_options)
    child1 = Vector{Any}(nothing, gp.numNodes)
    for i in 1:gp.numNodes
        if i in gp.nodesBelow[crossover_cut] || i == crossover_cut
            child1[i] = parent2[i]
        else
            child1[i] = parent1[i]
        end
    end
    return child1
end  

function stage_crossover(parent1, parent2, gp::GP)
    changeset = fill(1,gp.stages)
    while minimum(changeset) == maximum(changeset)
        for i in 1:gp.stages
            changeset[i] = rand(gp.rng,1:2)
        end
    end
    child1 = []; child2 = []
    for i in 1:gp.stages
        if changeset[i] == 1
            append!(child1,tuple(parent1[i])); append!(child2,tuple(parent2[i]))
        else
            append!(child1,tuple(parent2[i])); append!(child2,tuple(parent1[i]))
        end
    end
    return child1, child2
end

function get_depth(treeArray, gp::GP)
    return maximum(gp.nodeDepth[i] for i in 1:gp.numNodes if treeArray[i] !== nothing)
end

function get_expression_helper(treeArray, pos, children, binarytree)
    if children[pos] != []
        if binarytree c1, c2 = children[pos] else  c1, c2, c3 = children[pos] end
        if treeArray[c1] !== nothing
            if treeArray[c2] === nothing
                return Expr(:call, treeArray[pos], get_expression_helper(treeArray, c1, children, binarytree))
            else
                if binarytree
                    return Expr(:call, treeArray[pos], get_expression_helper(treeArray, c1, children, binarytree), get_expression_helper(treeArray, c2, children, binarytree))
                else
                    if treeArray[c3] === nothing
                        return Expr(:call, treeArray[pos], get_expression_helper(treeArray, c1, children, binarytree), get_expression_helper(treeArray, c2, children, binarytree))
                    else
                        return Expr(:call, treeArray[pos], get_expression_helper(treeArray, c1, children, binarytree), get_expression_helper(treeArray, c2, children, binarytree), get_expression_helper(treeArray, c3, children, binarytree))
                    end
                end
            end
        else
            return treeArray[pos]
        end
    else
        return treeArray[pos]
    end
end

function get_expression(treeArray, gp::GP)
    return get_expression_helper(treeArray,1,gp.children, gp.binarytree)
end

isnum(ex) = isa(ex, Number)
iszeronum(root) = isnum(root) && iszero(root)
isonenum(root) = isnum(root) && isone(root)
isdiv(ex) = ex in [/, div, pdiv, aq]
isexpr(ex) = isa(ex, Expr)
issym(ex) = isa(ex, Symbol)
isexprsym(ex) = isexpr(ex) || issym(ex)
isbinexpr(ex) = isexpr(ex) && length(ex.args) == 3

function evaluate(ex::Expr, vals) 
    exprm = ex.args
    exvals = (isexpr(nex) || issym(nex) ? evaluate(nex, vals) : nex for nex in exprm[2:end])
    exprm[1](exvals...)
end

function evaluate(ex::Symbol, vals)
    return vals[ex]
end 

function drawTree(tree_expr::Expr, stage=1, reduced = false, feature = nothing)
    plot(tree_expr, method=:tree, nodeshape=:rect, curves=false, fontsize=10, nodecolor="white", nodesize=0, shorten=0.15, size=(1000,800))
    if reduced == false
        savefig(plot!(), "C:\\Users\\floppy\\Desktop\\Tree_Plots\\treePlot$stage.svg")
        else
        savefig(plot!(), "C:\\Users\\floppy\\Desktop\\Tree_Plots\\treePlot$(stage)reducedBy$(feature).svg")
    end
end


function selection_helper(pop, gp::GP)
    # first crossover is always done with fitness value
    if !gp.markerdensity

        # TODO use sampling? instead of rand
        tournamentList = StatsBase.sample(gp.rng, pop, gp.tournamentSize, replace = false, ordered = true)   
        for i in 1:gp.tournamentSize
            if rand(gp.rng) < gp.tournamentProb
                return tournamentList[i]
            end
        end
        return tournamentList[end]
    else
        # just random selection
        return StatsBase.sample(gp.rng, pop)
        # TODO tournament based on pareto? not necessarily useful?
        # tournamentList = [rand(gp.rng,1:gp.popSize) for _ in 1:gp.tournamentSize]
        # sort_crowded!(tournamentList,pop)

        # for i in 1:gp.tournamentSize
        #     if rand(gp.rng) < gp.tournamentProb
        #         return tournamentList[i]
        #     end
        # end
        # return tournamentList[end]
    end
end

function selection_one(pop,gp::GP)
    return selection_helper(pop,gp)
end

function selection(pop, gp::GP)
    parent1 = selection_helper(pop, gp)
    foundPartner = false
    parent2 = selection_helper(pop,gp)
    while !foundPartner
        if parent1 != parent2
            foundPartner = true
            break
        end
        parent2 = selection_helper(pop,gp)
    end
    return parent1, parent2
end

function remove_dub(pop)
    return(unique!(pop))
end

function rand_new(number,gp::GP)
    tmpPop = []
    for _ in 1:number
        individuum = [] 
        for _ in 1:gp.stages
            append!(individuum,[build_tree(gp,rand(gp.rng) < gp.ratioGrow ? false : true, depthRegulator = sample(gp.rng,[i for i in gp.minDepth:gp.maxDepth]))])
        end

        # initialze with age 1
        append!(tmpPop, detF(individuum,1,gp))
        
    end

    # sort to improve algorithm in fitness only sorting?
    sort!(tmpPop, by=i->i[2])
    return tmpPop   
end

function next_gen!(gp::GP, pop, ngen)
    """ creates the next generation by performing
    selection, crossover and mutation
    """

    if !gp.stationary_samples
        if ngen % gp.change_every == 0
            if gp.sample_cycling
                gp.sample_tens_digit = (ngen ÷ gp.change_every) % (gp.steps_in_cycle)
            else
                setsamples!.(gp.envs, gp.rng, nSamples = gp.samples_generation)
            end
        end
    end

    next_gen = []
    # dynamically adapt? explore effects for this
    selectedMutationMethods = [1,2,3,4]
    # perform either crossover, mutation or reproduction
    count = 0
    targetsize = ifelse(gp.markerdensity, gp.popSize, gp.popSize - gp.elitism)
    while count < targetsize
        #draw method
        tmp = rand(gp.rng)
        child1 = []
        if tmp < gp.probCro
            #crossover
            parent1, parent2 = selection(pop, gp)

            if rand(gp.rng) < 0.5
                # two child method
                child2 = []
                for i in 1:gp.stages
                    tmp1, tmp2 = crossover(parent1[1][i], parent2[1][i], gp)
                    append!(child1, tuple(tmp1)); append!(child2, tuple(tmp2))
                end
                age = max(parent1[3],parent2[3]) +1
                append!(next_gen, detF(child1,age,gp)); append!(next_gen, detF(child2,age,gp))
                count += 2
            else
                # one child method
                for i in 1:gp.stages
                    tmp1 = crossover_one_child(parent1[1][i], parent2[1][i], gp)
                    append!(child1, tuple(tmp1))
                end
                age = max(parent1[3],parent2[3]) +1
                append!(next_gen, detF(child1,age,gp))
                count += 1
            end 
        elseif tmp < gp.probCro + gp.probMut
            # mutation
            m = sample(gp.rng, selectedMutationMethods)
            parent1 = selection_one(pop,gp)
            for i in 1:gp.stages
                tmp1 = mutation!(deepcopy(parent1[1][i]),gp,m)
                append!(child1,tuple(tmp1))
            end
            age = parent1[3] +1
            append!(next_gen, detF(child1,age,gp))
            count += 1
        else 
            #reproduction
            push!(child1,deepcopy(selection_one(pop,gp)))
            
            # increment age under conditions?
            child1[1][3] += 1

            # if changing scenarios via batches -> new evaluation
            append!(next_gen,child1)
            count += 1
        end
    end

    return next_gen
end
    
function new_pop(next_gen, gp::GP, pop, ngen)
    
    # get new fitness for old population if changed samples
    if !gp.stationary_samples
        if ngen % gp.change_every == 0
            for ind in pop
                ind[2] = 0
                fitness = 0
                for e in eachindex(gp.envs)
                    tmpF, tmpO = evalfitness([get_expression(ind[1][k],gp) for k in 1:gp.stages], gp.envs[e],
                        gp.objective, gp.samples_generation, gp.rng, gp.sample_tens_digit) #  ;type = "greedy") # -> sum(fitness),fitness, metrics
                    fitness += (tmpF / gp.samples_generation)
                end
                fitness /= length(gp.envs)
                ind[2] = fitness
            end
             if !gp.markerdensity sort!(pop, by=i->i[2]) end
        end
    end

    newPop = []
    if !gp.markerdensity
        # add elitism
        for i in 1:gp.elitism
            child1 = pop[i]
            # increment age
            child1[3] += 1
            # if changing scenarios via batches -> new evaluation
            push!(newPop, child1)
        end
        append!(newPop, next_gen)
        # remove dublicated
        unique!(newPop)
        # sort
        sort!(newPop, by=i->i[2])

        tmpL = length(newPop)
        tmpT = gp.popSize - gp.randomEach
        if tmpL >= tmpT
            newPop = newPop[1:tmpT]
            randomNew = rand_new(gp.randomEach,gp)
            append!(newPop,randomNew)
        else
            randomNew = rand_new(gp.popSize-tmpL,gp)
            append!(newPop,randomNew)
        end
    else
        # marker density sorting
        # combine new and old pop
        newPop = vcat(pop, next_gen)

        # assign marker density to population
        assign_marker_density!(newPop)

        # tournament selection for new pop until reach size == popSize
        while length(newPop) > (gp.popSize - gp.randomEach)
            tournamentlist = StatsBase.sample(newPop, gp.paretoTournamentSize, replace = false)
            pareto_tournament_removal!(newPop, tournamentlist, gp)
        end

        randomNew = rand_new(gp.popSize-length(newPop),gp)
        append!(newPop, randomNew)

    end
    @assert length(newPop) == gp.popSize
    sort!(newPop, by=i->i[2])
    return newPop
end

function tiebreaker(a, b, gp::GP)
    # compare size
    # TODO if multiple rules in an individuum
    sizea = sizeofrule(a[1][1])
    sizeb = sizeofrule(b[1][1])
    if sizea < sizeb
        return b
    elseif sizea > sizeb
        return a
    else
        if rand(gp.rng) < 0.5
            return a
        else
            return b
        end
    end
end

function sizeofrule(rule)
    return count(!isnothing, rule)
end
    
function pareto_tournament_removal!(pop, list, gp::GP)
    if length(list) == 2
        if list[1][2] == list[2][2] && list[1][5] == list[2][5]
            remove = tiebreaker(list[1], list[2], gp)
            index = findfirst(x -> x == remove, pop)
            deleteat!(pop, index)
        elseif (list[1][2] >= list[2][2] && list[1][5] >= list[2][5]) 
            deleteat!(pop, findfirst(x -> x == list[1], pop))
        elseif list[1][2] <= list[2][2] && list[1][5] <= list[2][5]
            deleteat!(pop, findfirst(x -> x == list[2], pop))
        end
    else
        dominated = []
        for i in list
            for j in list
                if i != j
                    if i[2] == j[2] && i[5] == j[5]
                        push!(dominated, tiebreaker(list[1], list[2], gp))
                    elseif i[2] >= j[2] && i[5] >= j[5]
                        push!(dominated, i)
                    elseif i[2] <= j[2] && i[5] <= j[5]
                        push!(dominated, j)
                    end
                end
            end
        end
        for i in unique(dominated)
            deleteat!(pop,findfirst(x -> x == i, pop))
        end
    end
end

function change_float!(treeArray, gp::GP)
    rand_nodes = shuffle([i for i in 1:gp.numNodes])
    for i in rand_nodes        
        if isa(treeArray[i], Number)
            orig = treeArray[i]
            treeArray[i] = rand(gp.rng,max(0.00001,orig-gp.ilsFloatRange):0.00001:min(1.0,orig+gp.ilsFloatRange))
            break
        end
    end
end

has_floats(treeArray) = any(isa(i,Number) for i in treeArray)

function convert_rule(rule)
    ruleDict = Dict()
    for i in eachindex(rule)
        if rule[i] in PRIMITIVES
            ruleDict[i] = FUN_STRING_DICT[rule[i]]
        elseif rule[i] in FEATURES
            ruleDict[i] = FEAT_STRING_DICT[rule[i]]
        elseif rule[i] !== nothing
            ruleDict[i] = rule[i]
        end
    end
    return ruleDict
end

function replaceValue(parent)
    replacement = 0
    if parent == add || parent == sub
        replacement = 0
    elseif parent == mul || parent == div
        replacement = 1
    elseif parent == max
        replacement = -Inf
    elseif parent == min
        replacement = Inf
    end
    return replacement
end

function reduceTree(TreeVector::Vector , feature::Symbol, gp::GP) 

    redVector = Vector{Any}(nothing, gp.numNodes)
    tabuList = []
    appearanceCount = 0

    for i in reverse(0:gp.maxDepth)
        for NrEle in gp.depthNodes[i]
            if TreeVector[NrEle] == feature
                appearanceCount = 1 
                tmp = NrEle
                while TreeVector[gp.parents[tmp]] in gp.primitives1 
                    tmp = gp.parents[tmp]
                    append!(tabuList,tmp)
                    if tmp == 1
                        return false
                    end
                end
                redVector[tmp] = replaceValue(TreeVector[gp.parents[tmp]])
            elseif NrEle ∉ tabuList
                redVector[NrEle] = deepcopy(TreeVector[NrEle])
            end
        end
    end
    if appearanceCount != 0
        return redVector
    else
        return false
    end
end

function dictReduced(TreeVector::Vector, gp::GP)
    dictRed = Dict{String, Array{Expr}}("baseline" => [get_expression(TreeVector, gp)])
    for f in gp.features
        tmpV = reduceTree(TreeVector,f,gp)
        if tmpV != false
            dictRed[string(f)] = [get_expression(tmpV,gp)]
        end
    end
    return dictRed
end

function featImp(TreeArray::Array,gp::GP)
    fImp = Dict{Any,Any}()
    for s in gp.stages
        dictRed = dictReduced(TreeArray[s],gp)
        fImp[s] = evaluate_reduced_trees(dictRed, PROBLEMS[1], SAMPLES[1][1])
    end
    return fImp
end


function detF(indi, age, gp::GP)

    fitness = 0
    objectives = [0,0,0,0,0,0]
    fitness_instance = []
    objective_instance = []
    for e in eachindex(gp.envs)
        tmpF, tmpO = evalfitness([get_expression(indi[k],gp) for k in 1:gp.stages], gp.envs[e],
            gp.objective, gp.samples_generation, gp.rng, gp.sample_tens_digit) #  ;type = "greedy") # -> sum(fitness),fitness, metrics
        fitness += (tmpF / gp.samples_generation)
        # if gp.nsga_2 != "n"
        #     if gp.nsga_2 == "o"
        #         objectives += tmpO
        #     elseif gp.nsga_2 == "i"
        #         append!(fitness_instance,tmpF)
        #     else
        #         deleteat!(tmpO,findall(x -> x == 0,OBJ_WEIGHTS))
        #         append!(objective_instance,tmpO)
        #     end
        # end
    end
    fitness /= length(gp.envs)

    if gp.markerdensity
        Ftuple = tuple([indi, fitness, age, get_marker(indi,gp)[1], 0]) # nothing as placeholder for marker; 0 as placeholder for density
    else
        Ftuple = tuple([indi, fitness, age])
    end
    return Ftuple
end

function get_marker(individum, gp::GP)
    marker = []
    for k in gp.stages
        push!(marker,string(get_marker_helper(individum[k], gp.markerstart, gp.children, gp.markerend, gp.nodeDepth)))
    end
    # TODO eliminate equivalent markers? e.g. 1+2 = 2+1 
    return marker
end

function get_marker_helper(treeArray, pos, children, endpoint, depth)
    if children[pos] != []
        c1,c2 = children[pos]
            if treeArray[c1] !== nothing && depth[c1] < endpoint
                if treeArray[c2] === nothing
                    return Expr(:call, treeArray[pos], get_marker_helper(treeArray, c1, children, endpoint, depth))
                else
                    return Expr(:call, treeArray[pos], get_marker_helper(treeArray, c1, children, endpoint, depth),
                                                        get_marker_helper(treeArray, c2, children, endpoint, depth))
                end
            else
                return treeArray[pos]
            end
    else
        return treeArray[pos]
    end
end

function assign_marker_density!(pop)
    occurencDict = Dict{String, Array{Int}}()
    for (i,individum) in enumerate(pop)
        # get marker
        marker = individum[4]  # TODO if multiple rules for each stage change this
        if marker in keys(occurencDict)
            append!(occurencDict[marker],i)
        else
            occurencDict[marker] = [i]
        end
    end
    for (marker,indicies) in occurencDict
        density = (length(indicies) / length(pop))
        for i in indicies
            pop[i][5] = density
        end
    end
end

function sortallsamples(pop,gp)
    nrEvals = gp.samples_generation * gp.steps_in_cycle
    for ind in pop
        ind[2] = 0
        fitness = 0
        for e in eachindex(gp.envs)
            tmpF, tmpO = evalfitness([get_expression(ind[1][k],gp) for k in 1:gp.stages], gp.envs[e],
                gp.objective, nrEvals, gp.rng, 0) #  ;type = "greedy") # -> sum(fitness),fitness, metrics
            fitness += (tmpF / gp.samples_generation)
        end
        fitness /= length(gp.envs)
        ind[2] = fitness
    end
    
    sort!(pop, by=i->i[2])
end





function simplify_expression(expr, gp)

    """
    Simplifies a mathematical expression represented as a binary tree, ensuring the resulting expression is 
    as simplified as possible by eliminating unnecessary operations like adding 0.0. 
    
    -------
    return: An optimized and simplified binary tree representing the mathematical expression, 
            where unnecessary operations are eliminated, and the tree structure is maintained.
    """

    featureEffects = Dict{Symbol, Int}()
    constantSum = 0.0

    function traverse(node, sign = 1)
        if isa(node, Number)
            constantSum += node * sign
        elseif isa(node, Symbol) && node in gp.features
            featureEffects[node] = get(featureEffects, node, 0) + sign
        elseif isa(node, Expr) && node.head == :call
            if node.args[1] == SchedulingRL.sub || node.args[1] == SchedulingRL.add
                for arg in node.args[2:end]
                    traverse(arg, node.args[1] == SchedulingRL.sub && arg !== node.args[2] ? -sign : sign)
                end
            end
        end
    end

    traverse(expr)

    # Rebuild the binary tree with simplification logic applied
    function construct_tree()
        if constantSum == 0.0 && isempty(featureEffects)
            return 0.0  # Return 0 if there are no features and the constant sum is 0
        end
        
        tree = nothing  # Initialize with nothing to indicate no tree has been constructed yet
        
        if constantSum != 0.0
            tree = constantSum
        end

        for (feature, count) in featureEffects
            for _ in 1:abs(count)
                if tree === nothing
                    tree = feature  # First feature becomes the tree if it's the first element
                else
                    operation = count > 0 ? SchedulingRL.add : SchedulingRL.sub
                    # If adding or subtracting 0, just use the existing tree to avoid unnecessary operations
                    if !(operation == SchedulingRL.add && tree == 0.0)
                        tree = Expr(:call, operation, tree, feature)
                    end
                end
            end
        end

        return tree === nothing ? 0.0 : tree  # Return 0 if no tree was constructed
    end

    return construct_tree()
end

function reduceTreeNumericals(expr)
    # Check if the expression directly contains a numerical value
    function containsDirectNumericValue(node)
        if isa(node, Expr)
            # Check if any of the direct arguments are numeric
            return any(arg -> isa(arg, Number), node.args[2:end])
        end
        return false
    end

    # Directly replace numerical values with 0 in the expression
    function directReplaceNumericWithZero(node)
        if isa(node, Expr) && containsDirectNumericValue(node)
            newArgs = map(arg -> isa(arg, Number) ? 0 : arg, node.args)
            return Expr(node.head, newArgs...)
        elseif isa(node, Expr)
            # Apply replacement to child nodes
            return Expr(node.head, node.args[1], directReplaceNumericWithZero.(node.args[2:end])...)
        else
            return node
        end
    end

    return directReplaceNumericWithZero(expr)
end

function analyze_tree(expr)
    #@warn "Only works for add and sub operations and apply implify_expression() and  reduceTreeNumericals() fist "
     # Initialize mappings and counters
     featureToPrimitives = Dict{Symbol, Symbol}()  # Each feature maps to a single operation
     numvalue = 0.0
     FeatureCounter = Dict{Symbol, Int}()
 
     # Recursive function to traverse the tree and analyze features
     function traverse(node, currentContext=:add)
         if isa(node, Expr) && node.head == :call
             # Determine if the current node represents a subtraction operation
             isSub = typeof(node.args[1]) === typeof(SchedulingRL.sub)
 
             # Determine the current context for child nodes
             if isSub
                 # For subtraction, the context is inverted for the right child
                 traverse(node.args[2], currentContext)  # Left child inherits the current context
                 traverse(node.args[3], currentContext == :add ? :sub : :add)  # Right child inverts context
             else
                 # For addition or other operations, maintain the current context
                 traverse(node.args[2], currentContext)
                 if length(node.args) > 2
                     traverse(node.args[3], currentContext)
                 end
             end
         elseif isa(node, Symbol)
             # Feature node encountered. Map the feature with its operation if not already mapped
             if !haskey(featureToPrimitives, node)
                 featureToPrimitives[node] = currentContext
                 FeatureCounter[node] = 1
             else
                 # Feature already encountered; increment occurrence counter
                 FeatureCounter[node] += 1
             end
         elseif isa(node, Number)
             # Number node encountered; adjust numvalue according to the current context
             adjustment = currentContext == :add ? node : -node
             numvalue += adjustment
         end
     end
 
     # Start the traversal from the root of the expression
     traverse(expr)
 
     return featureToPrimitives, numvalue, FeatureCounter
 end


function Covert_weights_from_GP(FeatureCounter::Dict, priofunnction_string::String)

    function extract_features_from_string(s::String)
        # Replace "-" with "+" to uniform the separators, then split, trim, and convert to Symbol
        features = Symbol.(strip.(split(replace(s, "-" => "+"), "+")))
        return features
    end

    features = extract_features_from_string(priofunnction_string)

    Initial_best_weights = Vector{Vector{Float32}}()
    weights_array = []
    for feature in features
        weight = FeatureCounter[Symbol(feature)]
        push!(weights_array, Float32(weight))
    end
    push!(Initial_best_weights, weights_array)
    return Initial_best_weights
end
