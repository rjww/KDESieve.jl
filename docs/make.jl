using Documenter, KDESieve

makedocs(;
    modules=[KDESieve],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/rjww/KDESieve.jl/blob/{commit}{path}#L{line}",
    sitename="KDESieve.jl",
    authors="Robert Woods",
    assets=String[],
)

deploydocs(;
    repo="github.com/rjww/KDESieve.jl",
)
