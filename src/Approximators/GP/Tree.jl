struct Tree
    node :: Union{Symbol, Function, Number}
    childs :: Array{Tree,1}
end

leaf(x::Union{Symbol, Number}) = Tree(x,[])

isLeaf(t::Tree) = t.childs == []

oneChild(t::Tree) = length(t.childs) == 1

tree(x::Union{Symbol, Function, Number}, ts:: Array{Tree,1}) = Tree(x,ts)

function drawTree(r::Tree)
    if (isLeaf(r))
        return string(r.node)
    elseif (oneChild(r))
        return string(r.node) * " -- " * drawTree(r.child[1])
    else
        xs = map( drawTree, r.childs )
        return string(r.node) * " -- " * "{ " * join(xs, ", ") * " }"
    end
end

function tree2Latex(t::Tree, fileName::String)
    s1 = "\\begin{tikzpicture}[every node/.style={circle, draw, minimum size=0.75cm}] \n\n "
    s2 = "\\graph [tree layout, grow=down, fresh nodes, level distance=0.5in, sibling distance=0.5in] \n {"
    s3 = " };\n \\end{tikzpicture}"
    s = s1 * s2 * drawTree(t) * s3
    outfile = open(fileName, "w")
    write(outfile, s)
    close(outfile)
end

