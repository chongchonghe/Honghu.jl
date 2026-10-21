# The version a release is published under is written in three places, and
# Zenodo takes it from CITATION.cff rather than from the tag: the three must
# agree, or the archived record carries the wrong version. RELEASING.md is the
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
