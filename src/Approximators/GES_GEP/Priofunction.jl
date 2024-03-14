struct Priorityfunction
    expression
    nn_mask::Vector{Bool}
    max::Vector{Float32}
    min::Vector{Float32}
    constant::Union{Float32, Nothing} # Optional field to store a numerical constant
end

function Priorityfunction(s::String,
                max::Union{Vector{Number},Number}=100,
                min::Union{Vector{Number},Number}=-100,
                constant::Union{Float32, Nothing}=nothing)

    features = ["PT", "DD", "RO", "RW", "ON", "JS", "ST", "NI", "NW", "SLA", "EW", "JWM","CW", "CJW", "CT","TT", "BSA", "DBA", "RF"]
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

    Priorityfunction(expr, mask , max , min , constant)
end


function PriorityFunctionFromTree(expr,
                                max::Union{Vector{Number},Number}=1,
                                min::Union{Vector{Number},Number}=-1,
                                keep_numerical_nodes::Bool=true, kwargs...)
    
    featureToPrimitives, numvalue, _ = analyze_tree(expr)

    plusOperations = []
    minusOperations = []

    # Separate features based on their operations
    for feature in keys(featureToPrimitives)
        operation = featureToPrimitives[feature]
        featureString = "$feature"
        
        if operation == :sub
            push!(minusOperations, "-$featureString")
        else
            push!(plusOperations, "+$featureString")
        end
    end

    # Concatenate operations, ensuring plus operations come first
    operationString = join(plusOperations, " ") * " " * join(minusOperations, " ")

    # Remove the leading "+" if present
    operationString = lstrip(operationString, '+')
    operationString = strip(operationString)
    operationString = String(operationString)

    println("Operation String generated from the tree: ", operationString, "\n")

    # Handle the constant
    constant = keep_numerical_nodes ? Float32(numvalue) : nothing

    return Priorityfunction(operationString,max,min, constant), operationString
    
end








