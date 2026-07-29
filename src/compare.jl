# Running one problem under several configurations and scoring the results.
#
# This is the operation the code exists for: "does variant B beat variant A, and
# does the answer survive refinement". It was already being written by hand --
# `docs/make_figures.jl` loops reconstruction schemes against resolutions and
# fits a slope, and the radiative-shock testset loops resolutions and checks
# that the error and the spike both converge -- so what follows is extraction,
# not invention.
#
# The table is written to disk and the plotting method reads it back. That is
# deliberate: a figure regenerated from a stored table is reproducible without
# re-running the simulations, and the provenance header records what produced
# the numbers.

"""
    ComparisonTable

The result of [`compare`](@ref): one row per (variant, resolution), a string
`variant` column and a numeric matrix.

Index it like a table -- `t[:l1_Trad]` gives a column, `t.variant` the labels --
or write it with [`write_comparison`](@ref) and read it back with
[`read_comparison`](@ref).
"""
struct ComparisonTable
    problem::String
    meta::Vector{Pair{String,String}}
    variant::Vector{String}
    columns::Vector{String}
    values::Matrix{Float64}
end

Base.size(t::ComparisonTable) = (length(t.variant), length(t.columns))
Base.isempty(t::ComparisonTable) = isempty(t.variant)

"""
    t[name] -> Vector{Float64}

The named numeric column of a [`ComparisonTable`](@ref).
"""
function Base.getindex(t::ComparisonTable, name::Union{Symbol,AbstractString})
    j = findfirst(==(String(name)), t.columns)
    j === nothing &&
        throw(ArgumentError("no column $name; have $(join(t.columns, ", "))"))
    return t.values[:, j]
end

"""
    t[name, variant] -> Vector{Float64}

The named column restricted to one variant and ordered by size, which is the
slice most analysis wants:

```julia
t[:l1_density, "HLLC"] ./ t[:l1_density, "LLF"]   # how much the solver bought
```
"""
function Base.getindex(t::ComparisonTable, name::Union{Symbol,AbstractString}, variant)
    v = String(variant)
    k = findall(==(v), t.variant)
    isempty(k) &&
        throw(ArgumentError("no variant $v; have $(join(variants(t), ", "))"))
    return t[name][k[sortperm(t[:size][k])]]
end

"""
    rows(t) -> Vector{NamedTuple}

The table as one `NamedTuple` per row, `variant` first. Convenient for filtering
and for feeding a plot.
"""
function rows(t::ComparisonTable)
    names = (:variant, Symbol.(t.columns)...)
    [NamedTuple{names}((t.variant[i], t.values[i, :]...)) for i in axes(t.values, 1)]
end

"""
    variants(t) -> Vector{String}

The distinct variant labels, in the order they were run.
"""
variants(t::ComparisonTable) = unique(t.variant)

# --- which columns are results ------------------------------------------------

"""
    BOOKKEEPING

Column names that record how a run went rather than how good it was. They are
worth storing beside the metrics -- a variant that wins only because it took ten
times as long should be visible -- but they are not scores, so they get no
convergence rate and no panel on a plot.
"""
const BOOKKEEPING = ("size", "cells", "steps", "wall_s", "floors",
                     "newton_mean", "newton_max", "newton_failures")

"""
    metric_columns(t) -> Vector{String}

The columns of a [`ComparisonTable`](@ref) that are scores, i.e. everything not
in [`BOOKKEEPING`](@ref).
"""
metric_columns(t::ComparisonTable) = filter(!in(BOOKKEEPING), t.columns)

_decreasing(y) = length(y) >= 2 && all(y[i+1] < y[i] for i in 1:length(y)-1)

# Row indices per variant, ordered by size, so "falls under refinement" means
# what it says even for a table that was not written in size order.
function _by_variant(t::ComparisonTable)
    sizes = t[:size]
    map(variants(t)) do v
        k = findall(==(v), t.variant)
        (v, k[sortperm(sizes[k])])
    end
end

"""
    is_error_metric(t, metric) -> Bool

Whether `metric` behaves like an error: positive everywhere, and falling under
refinement for at least one variant.

Not every metric is one. `radhydro_shock`'s `spike` converges *to* 4.26 rather
than to zero, so a log-log slope through it is a number with no meaning and a
log axis is the wrong axis. Printing a table and plotting one both ask this,
rather than each deciding for itself.
"""
function is_error_metric(t::ComparisonTable, metric)
    y = t[metric]
    all(>(0), y) || return false
    return any(_decreasing(y[k]) for (_, k) in _by_variant(t))
end

function Base.show(io::IO, ::MIME"text/plain", t::ComparisonTable)
    println(io, "ComparisonTable: ", t.problem, ", ", length(t.variant), " rows")
    for (k, v) in t.meta
        println(io, "  # ", k, " = ", v)
    end
    isempty(t) && return

    # A pure-hydro run has three all-zero Newton columns, and a run that needed
    # no floor repairs has a fourth. Four columns of zeros crowd out the metric
    # the table exists to show. They stay in the stored table -- "no repairs" is
    # a result worth keeping -- but here they collapse to one line.
    quiet = [c for (j, c) in enumerate(t.columns)
             if c in BOOKKEEPING && all(iszero, t.values[:, j])]
    keep = findall(!in(quiet), t.columns)

    w = max(7, maximum(length, t.variant; init=7))
    _row(io, w, "variant", t.columns[keep], t.columns[keep])
    for i in axes(t.values, 1)
        _row(io, w, t.variant[i], [_fmt(t.values[i, j]) for j in keep], t.columns[keep])
    end
    isempty(quiet) || println(io, "  (zero for every run: ", join(quiet, ", "), ")")
    _show_rates(io, t, w)
end

# One table line: a left-aligned label, then cells right-aligned under `headers`.
function _row(io::IO, w::Int, label, cells, headers)
    print(io, "  ", rpad(label, w))
    for (c, h) in zip(cells, headers)
        print(io, "  ", lpad(c, max(10, length(h))))
    end
    println(io)
end

# The slope is the reason to run a size sweep at all, so print it rather than
# making the caller fit it afterwards -- but only where it means something.
function _show_rates(io::IO, t::ComparisonTable, w::Int)
    ms = filter(m -> is_error_metric(t, m), metric_columns(t))
    bv = [p for p in _by_variant(t) if length(last(p)) >= 2]
    (isempty(ms) || isempty(bv)) && return
    println(io, "  convergence rate, -dlog(metric)/dlog(size):")
    _row(io, w, "variant", ms, ms)
    for (v, k) in bv
        cells = [_decreasing(t[m][k]) ? @sprintf("%.2f", convergence_rate(t, m; variant=v)) :
                 "--" for m in ms]
        _row(io, w, v, cells, ms)
    end
end

_fmt(v::Float64) = isinteger(v) && abs(v) < 1e15 ? string(Int(v)) : @sprintf("%.4g", v)

# --- running -----------------------------------------------------------------

# Variants may be given as a NamedTuple of NamedTuples, which labels itself, or
# as anything iterable of `label => kwargs` pairs when the labels need to be
# strings.
_variant_list(v::NamedTuple) = [String(k) => getfield(v, k) for k in keys(v)]
_variant_list(v::AbstractVector{<:Pair}) = [String(first(p)) => last(p) for p in v]
_variant_list(v) = [String(first(p)) => last(p) for p in v]

# Labels for a one-keyword sweep. `HLLC()` should label its column "HLLC": the
# type name *is* the scheme's name, and `string(HLLC())` would say "HLLC()".
# Numbers label themselves.
_name(v) = String(nameof(typeof(v)))
_name(v::Union{Number,Symbol,AbstractString}) = string(v)

# `PLM(MinMod())` and `PLM(VanLeer())` are both named "PLM", so fall back to the
# full type, which distinguishes them and stays short. Pass explicit labels via
# the general form if even that collides.
function _labels(values)
    ls = [_name(v) for v in values]
    return allunique(ls) ? ls : [string(typeof(v)) for v in values]
end

# Problem factories spell their resolution keyword `nx` (1D) or `n` (2D/3D).
# Ask the method rather than making the caller remember which.
function _size_key(f)
    try
        kws = Base.kwarg_decl(only(methods(f)))
        for k in (:nx, :n)
            k in kws && return k
        end
    catch
        # `kwarg_decl` is not a documented interface; if it ever moves, fall
        # back to the common case rather than failing the run.
    end
    return :nx
end

"""
    compare(problem, variants; sizes, metrics, tmax, path, outdir, verbose) -> ComparisonTable

Run `problem` once per (variant, size), score each run, and return the results
as a [`ComparisonTable`](@ref) which is also written to disk.

  - `problem` : a factory from [`Hydro3D.Problems`](@ref), or any callable
    returning `(sim, tmax)`.
  - `variants`: a `NamedTuple` of keyword `NamedTuple`s, which labels itself, or
    an iterable of `label => kwargs`. Each is splatted into `problem`.
  - `sizes`   : resolutions to sweep, default `(64, 128)`. Passed as the
    factory's `nx` or `n` keyword, whichever it declares; override with
    `size_key`.
  - `metrics` : `NamedTuple` of `metric(sim) -> Real`. Defaults to
    [`default_metrics`](@ref)`(problem)`.
  - `tmax`    : override the factory's suggested stopping time.
  - `path`    : where to write the table; defaults to `<problem>_compare.txt`
    inside `outdir` (default the working directory). Pass `path = nothing` to
    skip writing.

Every run also records `cells`, `steps`, `wall_s`, `floors` and the Newton
statistics, so a variant that is more accurate *because* it took ten times as
long, or that only converged with help from the positivity floors, is visible
rather than flattering.

```julia
using Hydro3D
const P = Hydro3D.Problems

t = compare(P.radhydro_shock,
            (P1 = (closure = EddingtonClosure(),),
             M1 = (closure = LevermoreClosure(),));
            sizes = (64, 128, 256))
```

Then `plotcompare(t)` (with Plots loaded), or `plotcompare("radhydro_shock_compare.txt")`
in a later session, redraws it without re-running anything.

See also the one-keyword form, [`compare(problem, key, values)`](@ref).
"""
function compare(problem, variants;
                 sizes=(64, 128),
                 metrics=nothing,
                 tmax=nothing,
                 size_key::Symbol=_size_key(problem),
                 path=:default,
                 outdir=".",
                 verbose::Bool=true)
    vs = _variant_list(variants)
    isempty(vs) && throw(ArgumentError("compare: no variants given"))
    # Duplicate labels would silently merge into one variant when the table is
    # sliced, and the merged rows would be fitted as if they were a size sweep.
    allunique(first.(vs)) ||
        throw(ArgumentError("compare: variant labels must be distinct, got " *
                            join(first.(vs), ", ")))
    ms = metrics === nothing ? default_metrics(problem) : metrics
    isempty(ms) && throw(ArgumentError("compare: no metrics given"))

    mnames = String.(collect(keys(ms)))
    columns = ["size", "cells", "steps", "wall_s", "floors",
               "newton_mean", "newton_max", "newton_failures", mnames...]
    label = Vector{String}()
    vals = Vector{Vector{Float64}}()

    verbose && @printf("compare: %s, %d variant(s) x %d size(s)\n",
                       _problem_name(problem), length(vs), length(sizes))
    for (name, kw) in vs
        _warmup(problem, size_key, minimum(sizes), tmax, kw)
        for sz in sizes
            sim, t_suggested = problem(; size_key => sz, kw...)
            t_end = tmax === nothing ? t_suggested : tmax
            el = time()
            run!(sim; tmax=t_end, verbose=false)
            el = time() - el

            d = sim.diag
            row = Float64[sz, prod(sim.grid.n), d.steps, el, nrepaired(sim),
                          d.newton_cells > 0 ? d.newton_iterations / d.newton_cells : 0.0,
                          d.newton_max, d.newton_failures]
            for g in values(ms)
                push!(row, float(g(sim)))
            end
            push!(label, name)
            push!(vals, row)
            if verbose
                @printf("  %-14s %-6d  %6.2f s  ", name, sz, el)
                for (m, mn) in zip(row[9:end], mnames)
                    @printf("%s = %-10.4g", mn, m)
                end
                println()
            end
        end
    end

    values_matrix = permutedims(reduce(hcat, vals))
    t = ComparisonTable(_problem_name(problem), _provenance(vs, sizes, mnames),
                        label, columns, values_matrix)

    p = path === :default ? joinpath(outdir, _problem_name(problem) * "_compare.txt") : path
    if p !== nothing
        mkpath(dirname(abspath(p)))
        write_comparison(t, p)
        verbose && println("wrote ", p)
    end
    return t
end

"""
    compare(problem, key::Symbol, values; kwargs...) -> ComparisonTable

Sweep a single keyword: run `problem` once per entry of `values`, passing it as
`key`, and label each run with the value.

```julia
compare(P.sod_shock_tube, :riemann, (HLLC(), LLF()); sizes = (64, 128, 256))
```

This is the common case -- one knob, several settings -- and it writes the
scheme's name once instead of three times. `HLLC()` labels its rows `HLLC`;
numbers label themselves. Use the general form when a variant needs more than
one keyword, or when you want the labels to say something else.

The default table is `<problem>_<key>_compare.txt`, so sweeping two different
knobs on the same problem gives two files. All other keywords are as for
[`compare`](@ref).
"""
function compare(problem, key::Symbol, values; path=:default, outdir=".", kwargs...)
    vs = [l => NamedTuple{(key,)}((v,)) for (l, v) in zip(_labels(values), values)]
    # Naming the key here matters: sweeping the solver and then the
    # reconstruction on one problem would otherwise write both tables to
    # `<problem>_compare.txt` and silently lose the first.
    p = path === :default ?
        joinpath(outdir, string(_problem_name(problem), "_", key, "_compare.txt")) : path
    return compare(problem, vs; path=p, outdir=outdir, kwargs...)
end

_problem_name(f) = String(nameof(f))

# Julia compiles on first call, and each variant produces its own concrete
# `Simulation` type, so the first timed run of every variant would otherwise be
# charged a few hundred milliseconds of compilation. Measured on the Sod tube,
# that made a 64-cell run look 550x slower than it is and slower than the same
# variant at 256 cells -- which inverts the one column people read to judge
# cost. So run each variant once first, at the smallest size, for one step, and
# throw the result away. Compiling is the point; the answer is not.
function _warmup(problem, size_key, sz, tmax, kw)
    sim, t_suggested = problem(; size_key => sz, kw...)
    run!(sim; tmax=(tmax === nothing ? t_suggested : tmax), maxsteps=1, verbose=false)
    return nothing
end

function _provenance(vs, sizes, mnames)
    [
        "variants" => join(first.(vs), ", "),
        "variant_kwargs" => join((string(first(p), " = ", last(p)) for p in vs), " | "),
        "sizes" => join(sizes, ", "),
        "metrics" => join(mnames, ", "),
        "date" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "code" => _code_version(),
        "julia" => string(VERSION),
        "threads" => string(Threads.nthreads()),
    ]
end

# Best effort: a table whose figure is regenerated months later should say what
# produced it, but a missing git checkout must not fail the run.
function _code_version()
    dir = pkgdir(@__MODULE__)
    dir === nothing && return "unknown"
    try
        h = readchomp(`git -C $dir rev-parse --short HEAD`)
        dirty = !isempty(readchomp(`git -C $dir status --porcelain`))
        return dirty ? h * "-dirty" : h
    catch
        return "unknown"
    end
end

# --- storage -----------------------------------------------------------------

"""
    write_comparison(table, path) -> path

Write a [`ComparisonTable`](@ref) as a tab-separated text table with a `#`
comment header carrying the provenance. Readable by hand, by
[`read_comparison`](@ref), and by anything that skips comment lines.
"""
function write_comparison(t::ComparisonTable, path::AbstractString)
    open(path, "w") do f
        println(f, "# hydro3d comparison")
        println(f, "# problem = ", t.problem)
        for (k, v) in t.meta
            println(f, "# ", k, " = ", v)
        end
        println(f, "variant\t", join(t.columns, "\t"))
        for i in axes(t.values, 1)
            print(f, t.variant[i])
            for j in axes(t.values, 2)
                print(f, "\t", _store(t.values[i, j]))
            end
            println(f)
        end
    end
    return path
end

# Julia's default Float64 printing is the shortest decimal that parses back to
# the same bits, so the stored table is both readable and exactly recoverable.
# A fixed `%.Ng` would have to choose between those two.
_store(v::Float64) = isinteger(v) && abs(v) < 1e15 ? string(Int(v)) : string(v)

"""
    read_comparison(path) -> ComparisonTable

Read back a table written by [`write_comparison`](@ref). Needs nothing beyond
the package, so a stored comparison can be re-plotted or re-analysed in a
session that never loads a solver.
"""
function read_comparison(path::AbstractString)
    problem = "unknown"
    meta = Pair{String,String}[]
    columns = String[]
    label = String[]
    vals = Vector{Vector{Float64}}()
    for ln in eachline(path)
        isempty(strip(ln)) && continue
        if startswith(ln, '#')
            body = strip(ln[2:end])
            body == "hydro3d comparison" && continue
            i = findfirst('=', body)
            i === nothing && continue
            k, v = strip(body[1:i-1]), strip(body[i+1:end])
            k == "problem" ? (problem = String(v)) : push!(meta, String(k) => String(v))
        elseif isempty(columns)
            f = split(ln, '\t')
            first(f) == "variant" ||
                throw(ArgumentError("$path: expected a header row starting with `variant`"))
            columns = String.(f[2:end])
        else
            f = split(ln, '\t')
            length(f) == length(columns) + 1 ||
                throw(ArgumentError("$path: row has $(length(f)-1) values for $(length(columns)) columns"))
            push!(label, String(f[1]))
            push!(vals, [parse(Float64, x) for x in f[2:end]])
        end
    end
    isempty(columns) && throw(ArgumentError("$path: no header row found"))
    m = isempty(vals) ? zeros(0, length(columns)) : permutedims(reduce(hcat, vals))
    return ComparisonTable(problem, meta, label, columns, m)
end

"""
    convergence_rate(table, metric; variant = nothing) -> Real

Least-squares slope of `log(metric)` against `log(size)` for one variant,
negated, so that a second-order scheme returns about `2`.

With one variant in the table `variant` may be omitted.
"""
function convergence_rate(t::ComparisonTable, metric; variant=nothing)
    vs = variants(t)
    v = variant === nothing ? (length(vs) == 1 ? only(vs) :
                               throw(ArgumentError("table has $(length(vs)) variants; name one"))) :
        String(variant)
    k = findall(==(v), t.variant)
    isempty(k) && throw(ArgumentError("no variant $v; have $(join(vs, ", "))"))
    length(k) >= 2 ||
        throw(ArgumentError("need at least two sizes to fit a rate, got $(length(k))"))
    x = log.(t[:size][k])
    y = log.(t[metric][k])
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    return -sum((x .- xm) .* (y .- ym)) / sum(abs2, x .- xm)
end
