# The version of a release is written in Project.toml, CITATION.cff and the
# changelog, and again in the tag. Zenodo records the tag name as the version
# (v0.1.0 was archived as `v0.1.0`), while GitHub's "Cite this repository" reads
# CITATION.cff: the files must agree, so that the package, the citation GitHub
# offers and the archived record all name one version. RELEASING.md is the
# procedure this guards.

using TOML

@testset "release metadata agrees" begin
    root = normpath(joinpath(@__DIR__, ".."))
    version = TOML.parsefile(joinpath(root, "Project.toml"))["version"]
    cff = read(joinpath(root, "CITATION.cff"), String)
    m = match(r"(?m)^version:\s*\"?([^\s\"]+)\"?\s*$", cff)
    @test m !== nothing && m[1] == version
    @test occursin("## [$version]", read(joinpath(root, "CHANGELOG.md"), String))
end
