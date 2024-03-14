using SchedulingRL
using Documenter

DocMeta.setdocmeta!(SchedulingRL, :DocTestSetup, :(using SchedulingRL); recursive=true)

makedocs(;
    modules=[SchedulingRL],
    authors="ga36wab <jan.doerr@tum.de> and contributors",
    repo="https://gitlab.com/Jan-Niklas Dörr/SchedulingRL.jl/blob/{commit}{path}#{line}",
    sitename="SchedulingRL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Jan-Niklas Dörr.gitlab.io/SchedulingRL.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
