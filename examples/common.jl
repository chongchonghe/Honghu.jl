# Shared set-up for the example scripts.
#
# Figures are written by default whenever Plots can be loaded, because an
# example you have to post-process by hand is not much of an example. The
# escape hatches are for batch runs:
#
#   --no-plot        skip the figure, keep the data files
#   --outdir=DIR     where figures and data go (default: the working directory)
#
# Anything not starting with `--` is left in `OPTS` for the script to read as a
# positional argument.

# GR needs this to render off-screen; respect an existing setting.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

const NOPLOT = "--no-plot" in ARGS
const OPTS = filter(a -> !startswith(a, "--"), ARGS)
const OUTDIR = let i = findfirst(a -> startswith(a, "--outdir="), ARGS)
    i === nothing ? pwd() : ARGS[i][length("--outdir=")+1:end]
end

"""
Whether figures will be written: true when Plots loaded and `--no-plot` was not
given. Scripts should guard their plotting blocks with this.
"""
const PLOTTING = if NOPLOT
    false
else
    try
        @eval using Plots
        @eval default(fontfamily="sans-serif", framestyle=:box, grid=true,
                      gridalpha=0.15, dpi=140, legendfontsize=8,
                      titlefontsize=10, guidefontsize=9)
        true
    catch
        false
    end
end

"Path inside the output directory, creating it if needed."
function outpath(name)
    mkpath(OUTDIR)
    joinpath(OUTDIR, name)
end

"Save a figure and report where it went. A no-op when plotting is off."
function saveplot(p, name)
    PLOTTING || return nothing
    path = outpath(name)
    savefig(p, path)
    println("  wrote $path")
    return path
end

"Explain once, at the end of a script, why no figure appeared."
function plotting_note()
    PLOTTING && return nothing
    if NOPLOT
        println("  (--no-plot given, so no figure was written)")
    else
        println("  (no figure: Plots is not available. Install it with")
        println("     julia -e 'using Pkg; Pkg.add(\"Plots\")'")
        println("   and rerun; Honghu's plotting methods load automatically.)")
    end
    return nothing
end
