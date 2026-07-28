# Error norms, and the per-problem metric sets that `compare` scores runs with.
#
# The norms live here rather than being written out at each call site because
# they were written out at each call site: the relative L1 in the radiation
# temperature appeared byte-identically in the radiative-shock testset and in
# `docs/make_figures.jl`, with a third variant for smooth advection. A metric
# that is spelled out three times is a metric that will eventually disagree with
# itself.

"""
    relative_l1(num, ref) -> Real

Relative `L1` error `Σ|num - ref| / Σ|ref|` between a numerical result and a
reference of the same length.

This is the norm Quokka reports for its radiation tests, so a number produced
here is directly comparable with one produced there.
"""
function relative_l1(num, ref)
    length(num) == length(ref) ||
        throw(DimensionMismatch("relative_l1: $(length(num)) values against $(length(ref)) reference points"))
    den = sum(abs, ref)
    den > 0 || throw(ArgumentError("relative_l1: the reference is identically zero"))
    return sum(abs, num .- ref) / den
end

"""
    relative_l2(num, ref) -> Real

Relative `L2` error `√(Σ(num - ref)²/Σ ref²)`. Use when you care about a
localised feature that an `L1` norm would average away.
"""
function relative_l2(num, ref)
    length(num) == length(ref) ||
        throw(DimensionMismatch("relative_l2: $(length(num)) values against $(length(ref)) reference points"))
    den = sum(abs2, ref)
    den > 0 || throw(ArgumentError("relative_l2: the reference is identically zero"))
    return sqrt(sum(abs2, num .- ref) / den)
end

"""
    default_metrics(problem) -> NamedTuple

The metrics a built-in problem scores itself with, as a `NamedTuple` of
callables `metric(sim) -> Real` evaluated on the finished simulation.
[`compare`](@ref) uses these when you do not pass `metrics` yourself, so

```julia
compare(Hydro3D.Problems.radhydro_shock, variants; nx = (64, 128, 256))
```

needs no configuration at all.

Defined for the problems that have a reference solution to be scored against.
Problems without one (Kelvin-Helmholtz, the Marshak wave) have no default, and
`compare` then requires explicit `metrics`.

Adding a problem with an exact solution? Give it a method here, in one line, and
the test suite, the verification figures and the comparison harness all pick up
the same definition of "the error".
"""
function default_metrics end

default_metrics(f) = throw(ArgumentError(
    "$(nameof(f)) has no default metrics (it has no reference solution to score " *
    "against); pass `metrics = (; name = sim -> ...)` explicitly"))

# The informative error above is itself a method on `Any`, so `hasmethod` would
# answer `true` for every problem. Compare against that method instead.
const _NO_DEFAULT_METRICS = which(default_metrics, Tuple{Any})

"""
    has_default_metrics(problem) -> Bool

Whether [`default_metrics`](@ref) has a method for `problem`, as opposed to
falling through to the error.
"""
has_default_metrics(f) = which(default_metrics, Tuple{typeof(f)}) !== _NO_DEFAULT_METRICS

"""
    line(sim, name) -> Vector

The named field along the single active direction of a one-dimensional problem,
as a plain vector. A convenience for writing metrics.
"""
function line(sim::Simulation, name::Symbol)
    d = findfirst(sim.grid.active)
    d === nothing && throw(ArgumentError("line: the grid has no active direction"))
    count(sim.grid.active) == 1 ||
        throw(ArgumentError("line: the grid is $(ndim(sim.grid))D; `line` needs a 1D problem"))
    data, _ = field(sim, name)
    return vec(data)
end
