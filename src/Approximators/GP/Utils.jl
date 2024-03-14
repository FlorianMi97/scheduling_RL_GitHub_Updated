struct Feature
    name::String
end

function symbols(ex::Expr)
    syms = Symbol[]
    for e in ex.args
        isa(e, Symbol) && push!(syms, e)
        isa(e, Expr) && append!(syms, symbols(e))
    end
    unique!(syms)
end


latex(io::IO, root::Number; digits=3) = print(io, round(root, digits=digits))
latex(io::IO, root::Symbol; kwargs...) = print(io, root)
function latex(io::IO, root::Expr; digits=3)
    root.head != :call && print(io, expr)
    if root.args[1] == (/)
        print(io, "\\frac{")
        latex(io, root.args[2], digits=digits)
        print(io, "}{")
        latex(io, root.args[3], digits=digits)
        print(io, "}")
    elseif root.args[1] ∈ [+, -, *]
        print(io, "\\left(")
        latex(io, root.args[2], digits=digits)
        print(io, root.args[1])
        latex(io, root.args[3], digits=digits)
        print(io, "\\right)")
    elseif root.args[1] == (^)
        print(io, "{")
        latex(io, root.args[2], digits=digits)
        print(io, "}^{")
        latex(io, root.args[3], digits=digits)
        print(io, "}")
    else
        print(io, "\\")
        print(io, root.args[1])
        print(io, "\\left(")
        for ex in root.args[2:end]
            latex(io, ex, digits=digits)
        end
        print(io, "\\right)")
    end
end
# show(io::IO, ::MIME"text/latex", e::Expression) = latex(io, e.expr)

function infix(io::IO, root; digits=3)
    if isa(root, Number)
        print(io, round(root, digits=digits))
    elseif isa(root, Expr) && root.head == :call
        if root.args[1] ∈ [+, -, *, /, ^]
            print(io, "(")
            infix(io, root.args[2])
            infix(io, root.args[1])
            length(root.args)>2 && infix(io, root.args[3])
            print(io, ")")
        else
            infix(io, root.args[1])
            print(io, "(")
            for (i, ex) in enumerate(root.args[2:end])
                i > 1 && print(io, ", ")
                infix(io, ex)
            end
            print(io, ")")
        end
    else
        print(io, root)
    end
end
# show(io::IO, ::MIME"text/html", e::Expression) = infix(io, e.expr)