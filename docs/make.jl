# Build the documentation site: reference pages written from the docstrings, the
# Quarto render, a Markdown twin of every page, and a strict link check.
#
#     julia --project=. docs/make.jl                  # full build into docs/_site
#     julia --project=. docs/make.jl --generate-only  # pages only, then `quarto preview docs`
#
# Every page exists twice, built from one source in one run: the HTML Quarto
# renders, and the source itself, published beside it as `<page>.md` -- `.md`
# rather than `.qmd` because GitHub Pages serves `.qmd` as a download and `.md`
# as text. `honghu-docs.md` concatenates all of them and `llms.txt` indexes
# them. Quarto's own `llms-txt` is not used: in 1.9 it garbles math.
#
# The build fails on anything a reader would trip over: a `[`name`](@ref)` with
# no docstring behind it, a source file with no reference page, a Quarto
# warning, or a relative link or anchor in the output that does not resolve.
# Quarto has no strict mode of its own -- it only warns about missing `.qmd`
# targets -- so the check is done here, on the rendered site.

using Honghu

const ROOT = dirname(@__DIR__)
const DOCS = @__DIR__
const SRC = joinpath(ROOT, "src")
const SITE = joinpath(DOCS, "_site")
const REPO = "https://github.com/chongchonghe/Honghu.jl"
const BUNDLE = "honghu-docs.md"

# Reference page => (title, the files under src/ whose docstrings it holds). A
# trailing "/" takes a whole directory. A file with a docstring and no page fails
# the build, so a new file cannot drop its docstrings silently.
const PAGES = [
    "simulation" => ("Simulation and grid",
                     ["simulation.jl", "grid.jl", "variables.jl", "boundary.jl", "integrators.jl"]),
    "hydro" => ("Hydrodynamics",
                ["eos.jl", "hydro_system.jl", "limiters.jl", "reconstruction.jl", "riemann/"]),
    "radiation" => ("Radiation transport",
                    ["radiation_system.jl", "riemann_radiation.jl", "planck.jl", "opacity.jl"]),
    "rsla" => ("Reduced speed of light", ["rsla.jl"]),
    "coupling" => ("Matter-radiation coupling",
                   ["coupling.jl", "coupling_solvers.jl", "coupling_ode.jl", "source_terms.jl",
                    "dust.jl", "newton.jl", "rootfind.jl"]),
    "output" => ("Output, plotting and comparison",
                 ["output.jl", "plotting.jl", "metrics.jl", "compare.jl"]),
    "problems" => ("Test problems", ["problems/"]),
    "utilities" => ("Utilities and constants", ["staticvec.jl", "constants.jl"]),
]

# Hand-written pages, in navigation order after the reference.
const NOTES = ["coupling-new-method.md", "coupling-new-method-dust.md",
               "coupling-findings.md", "rsla-findings.md", "documentation.md"]

const MODULES = (Honghu, Honghu.Problems, Honghu.CGS)

# --- collecting the docstrings ------------------------------------------------

struct Entry
    name::String                              # as written in the heading
    id::String                                # anchor, unique across the site
    page::String
    public::Bool
    docs::Vector{Tuple{String,String,Int}}    # (text, path under the repo, line)
end

slug(name) = replace(name, "!" => "-bang", r"[^A-Za-z0-9_-]" => "-")

function page_of(path)
    rel = relpath(path, SRC)
    rel == "Honghu.jl" && return "index"
    for (page, (_, files)) in PAGES, f in files
        (rel == f || (endswith(f, "/") && startswith(rel, f))) && return page
    end
    return nothing
end

# The order files are listed in PAGES, then line: the order a reader of the
# source would meet them.
function file_rank(path)
    rel = relpath(path, SRC)
    n = 0
    for (_, (_, files)) in PAGES, f in files
        n += 1
        (rel == f || (endswith(f, "/") && startswith(rel, f))) && return n
    end
    return 0
end

function collect_entries()
    entries = Entry[]
    moddocs = Dict{Module,String}()
    unmapped = String[]
    for mod in MODULES, (b, multi) in Docs.meta(mod)
        docs = [(join(d.text), relpath(d.data[:path], ROOT), d.data[:linenumber])
                for d in (multi.docs[s] for s in multi.order)]
        sort!(docs; by=x -> (file_rank(joinpath(ROOT, x[2])), x[3]))
        page = page_of(joinpath(ROOT, docs[1][2]))
        page === nothing && (push!(unmapped, docs[1][2]); continue)
        if b.mod === mod && b.var === nameof(mod)     # the module's own docstring
            mod === Honghu && (moddocs[mod] = docs[1][1]; continue)
            name, public = "Honghu." * string(nameof(mod)), true
        elseif b.mod !== mod                           # a method on a Base function
            name, public = string(b.mod) * "." * string(b.var), false
        else
            # `CGS` exports nothing by design: its constants are reached as
            # `CGS.name`, and are the module's whole point.
            name = mod === Honghu.CGS ? "CGS." * string(b.var) : string(b.var)
            public = Base.isexported(mod, b.var) || mod === Honghu.CGS
        end
        push!(entries, Entry(name, slug(name), page, public, docs))
    end
    isempty(unmapped) ||
        error("no reference page for these files; add them to PAGES:\n  " *
              join(unique(unmapped), "\n  "))
    ids = [e.id for e in entries]
    dup = unique(filter(i -> count(==(i), ids) > 1, ids))
    isempty(dup) || error("duplicate anchors: $dup")
    # Module docstrings first, then file and line order.
    sort!(entries; by=e -> (!startswith(e.name, "Honghu."), file_rank(joinpath(ROOT, e.docs[1][2])),
                            e.docs[1][3]))
    return entries, moddocs
end

# A `[`X`](@ref)` may name `X` as `Honghu.X`, `CGS.X` or `X(args)`; all mean
# the entry `X`.
function build_refmap(entries)
    map = Dict{String,Entry}(e.name => e for e in entries)
    for e in entries
        short = last(split(e.name, '.'))
        haskey(map, short) || (map[short] = e)
    end
    return map
end

function resolve(refmap, target)
    t = replace(strip(target), r"\(.*\)$" => "")
    for k in (t, replace(t, r"^Honghu\." => ""), last(split(t, '.')))
        haskey(refmap, k) && return refmap[k]
    end
    return nothing
end

# --- docstring to Quarto markdown --------------------------------------------

const CALLOUT = Dict("note" => "note", "info" => "note", "compat" => "note",
                     "tip" => "tip", "warning" => "warning", "danger" => "important")

# The docstring as written, with five changes: the signature becomes a `julia`
# fence, `@ref` links point at their entry, headers drop below the entry heading
# (`demote` levels), admonitions become Quarto callouts, and `jldoctest` becomes
# `julia`. Formulas, Unicode and layout are left exactly as the author wrote them.
function convert_doc(text, page, refmap, unresolved; demote=3)
    lines = split(rstrip(text), '\n')
    out = String[]
    i = 1
    if !isempty(lines) && startswith(lines[1], "    ")
        push!(out, "```julia")
        while i <= length(lines) && startswith(lines[i], "    ")
            push!(out, lines[i][5:end]); i += 1
        end
        push!(out, "```")
    end
    infence = false
    callout = false
    while i <= length(lines)
        l = String(lines[i]); i += 1
        if callout && !(isempty(strip(l)) || startswith(l, "    "))
            push!(out, ":::", ""); callout = false
        end
        if occursin(r"^\s*```", l)
            infence = !infence
            l = replace(l, "```jldoctest" => "```julia")
        elseif !infence
            if (m = match(r"^(#{1,6})\s+(.*)$", l)) !== nothing
                l = "#"^min(length(m[1]) + demote, 6) * " " * m[2]
                push!(out, l); i <= length(lines) && !isempty(strip(lines[i])) && push!(out, "")
                continue
            elseif (m = match(r"^!!! (\w+)(?: \"(.*)\")?\s*$", l)) !== nothing
                kind = get(CALLOUT, m[1], "note")
                title = m[2] === nothing ? "" : " title=\"" * replace(m[2], "\"" => "'") * "\""
                push!(out, "::: {.callout-$kind$title}"); callout = true
                continue
            end
            callout && startswith(l, "    ") && (l = l[5:end])
            l = replace(l, r"\[`([^`]+)`\]\(@ref\)" => function (s)
                target = match(r"\[`([^`]+)`\]", s)[1]
                e = resolve(refmap, target)
                e === nothing && (push!(unresolved, target); return s)
                href = e.page == page ? "#$(e.id)" : "$(e.page).qmd#$(e.id)"
                return "[`$target`]($href)"
            end)
            occursin("(@ref", l) && push!(unresolved, "unsupported @ref form: " * strip(l))
        end
        push!(out, l)
    end
    callout && push!(out, ":::")
    return join(out, '\n')
end

srclink(path, line, sha) = "Source: [`$path:$line`]($REPO/blob/$sha/$path#L$line)"

function write_entry(io, e, refmap, unresolved, sha)
    println(io, "### `", e.name, "` {#", e.id, "}\n")
    for (text, path, line) in e.docs
        println(io, srclink(path, line, sha), "\n")
        println(io, convert_doc(text, e.page, refmap, unresolved), "\n")
    end
end

function write_reference(entries, moddocs, refmap, sha)
    dir = joinpath(DOCS, "reference")
    rm(dir; recursive=true, force=true)
    mkpath(dir)
    unresolved = String[]
    for (page, (title, files)) in PAGES
        open(joinpath(dir, page * ".qmd"), "w") do io
            println(io, "---\ntitle: \"", title, "\"\n---\n")
            println(io, "Docstrings from ", join(("`src/$f`" for f in files), ", "), ".\n")
            here = filter(e -> e.page == page, entries)
            for (heading, sel) in (("Exported", true), ("Internals", false))
                part = filter(e -> e.public == sel, here)
                isempty(part) && continue
                println(io, "## ", heading, "\n")
                foreach(e -> write_entry(io, e, refmap, unresolved, sha), part)
            end
        end
    end
    open(joinpath(dir, "index.qmd"), "w") do io
        println(io, "---\ntitle: \"Reference\"\n---\n")
        println(io, convert_doc(moddocs[Honghu], "index", refmap, unresolved; demote=1), "\n")
        println(io, "## Pages\n")
        for (page, (title, _)) in PAGES
            pub = [e for e in entries if e.page == page && e.public]
            print(io, "- [", title, "](", page, ".qmd)")
            isempty(pub) || print(io, ": ", join(("[`$(e.name)`]($page.qmd#$(e.id))" for e in pub), ", "))
            println(io)
        end
    end
    isempty(unresolved) ||
        error("@ref links with no docstring behind them:\n  " * join(unique(unresolved), "\n  "))
end

# The README is the front page. Its links are written for GitHub's view of the
# repository root and are rewritten for the site's view from docs/.
function write_index()
    s = read(joinpath(ROOT, "README.md"), String)
    s = replace(s, r"\A# Honghu\.jl\s*\n" => "")
    s = replace(s, "](docs/figures/" => "](figures/", "src=\"docs/figures/" => "src=\"figures/",
                "](../Hydro.jl)" => "](https://github.com/chongchonghe/Hydro.jl)",
                "](future.md)" => "]($REPO/blob/main/future.md)")
    write(joinpath(DOCS, "index.qmd"), "---\ntitle: \"Honghu.jl\"\n---\n\n" * s)
end

# --- render, twins, bundle ----------------------------------------------------

# Every rendered source, in navigation order, as a path under docs/.
sources() = ["index.qmd"; "reference/index.qmd"; ["reference/$p.qmd" for (p, _) in PAGES]; NOTES]

function render()
    rm(SITE; recursive=true, force=true)
    buf = IOBuffer()
    proc = run(pipeline(ignorestatus(`quarto render $DOCS`); stdout=buf, stderr=buf))
    log = String(take!(buf))
    print(log)
    success(proc) || error("quarto render failed")
    # Quarto colours its output, so the escape codes go before the match.
    warns = filter(l -> occursin(r"^\s*WARN", l), replace.(split(log, '\n'), r"\e\[[0-9;]*m" => ""))
    isempty(warns) || error("quarto warned:\n  " * join(warns, "\n  "))
end

mdpath(src) = splitext(src)[1] * ".md"
qmd_to_md(s) = replace(s, r"\]\((?!https?:)([^)\s#]+)\.qmd((?:#[^)\s]*)?)\)" => s"](\1.md\2)")

# The bundle sits at the site root, so a page's relative links are re-based
# there, and a link to another page's entry becomes a link to that entry.
function rebase_links(body, srcdir)
    replace(body, r"\]\((?!https?:)([^)\s#]+)((?:#[^)\s]*)?)\)" => function (m)
        path, frag = match(r"\]\(([^)\s#]+)((?:#[^)\s]*)?)\)", m).captures
        endswith(path, ".qmd") && !isempty(frag) && return "]($frag)"
        return "](" * qmd_to_md_path(normpath(joinpath(srcdir, path))) * frag * ")"
    end)
end
qmd_to_md_path(p) = endswith(p, ".qmd") ? p[1:end-4] * ".md" : p

function page_title(s)
    m = match(r"\A---\ntitle: \"(.*)\"\n---\n", s)
    m !== nothing && return m[1]
    return match(r"^# (.*)$"m, s)[1]
end

function write_twins_and_bundle(sha)
    bundle = IOBuffer()
    println(bundle, "# Honghu.jl documentation, all pages\n")
    println(bundle, "Every page of the documentation site in one file, generated from ",
            "[the repository]($REPO) at commit `$sha`. Each page below is also published ",
            "on its own, as HTML and as Markdown.\n")
    titles = String[]
    for src in sources()
        s = read(joinpath(DOCS, src), String)
        dst = joinpath(SITE, mdpath(src))
        mkpath(dirname(dst))
        write(dst, qmd_to_md(s))
        title = page_title(s)
        push!(titles, title)
        body = replace(s, r"\A---\n.*?\n---\n"s => "# $title\n")
        body = rebase_links(body, dirname(src))
        println(bundle, "\n---\n\n<!-- page: ", mdpath(src), " -->\n\n", body)
    end
    write(joinpath(SITE, BUNDLE), String(take!(bundle)))

    open(joinpath(SITE, "llms.txt"), "w") do io
        println(io, "# Honghu.jl\n")
        println(io, "> A three-dimensional radiation-hydrodynamics code in pure Julia: the Euler ",
                "equations and two-moment (M1) radiation, grey or multigroup, with an IMEX ",
                "PD-ARS integrator. The documentation, page by page, in Markdown.\n")
        println(io, "- [All pages in one file](", BUNDLE, ")\n")
        groups = (("Overview", 1:1), ("Reference", 2:length(PAGES)+2),
                  ("Notes", length(PAGES)+3:length(sources())))
        for (heading, r) in groups
            println(io, "## ", heading, "\n")
            for k in r
                println(io, "- [", titles[k], "](", mdpath(sources()[k]), ")")
            end
            println(io)
        end
    end
end

# --- the strict check ---------------------------------------------------------

const EXTERNAL = r"^(?:[a-z]+:|//)"

# Every relative href/src in the rendered HTML must name a file in the site, and
# every #fragment an id in that file.
function check_html()
    bad = String[]
    idcache = Dict{String,Set{String}}()
    ids(f) = get!(() -> Set(m[1] for m in eachmatch(r"\bid=\"([^\"]+)\"", read(f, String))),
                  idcache, f)
    for (dir, _, files) in walkdir(SITE), f in files
        endswith(f, ".html") || continue
        occursin("site_libs", dir) && continue
        file = joinpath(dir, f)
        for m in eachmatch(r"\b(?:href|src)=\"([^\"]*)\"", read(file, String))
            link = m[1]
            (isempty(link) || occursin(EXTERNAL, link)) && continue
            path, frag = occursin('#', link) ? split(link, '#'; limit=2) : (link, "")
            path = replace(String(path), r"\?.*$" => "", "%20" => " ")
            target = isempty(path) ? file : normpath(joinpath(dir, path))
            isdir(target) && (target = joinpath(target, "index.html"))
            if !isfile(target)
                push!(bad, "$(relpath(file, SITE)): $link")
            elseif !isempty(frag) && endswith(target, ".html") && !(frag in ids(target))
                push!(bad, "$(relpath(file, SITE)): $link (no such anchor)")
            end
        end
    end
    return bad
end

# Every relative link in the Markdown twins and the bundle must name a file in
# the site. Anchors are checked on the HTML, which is rendered from the same
# sources. Code is stripped first, so a `[x](y)` inside a code span is not
# mistaken for a link.
function check_markdown()
    bad = String[]
    for (dir, _, files) in walkdir(SITE), f in files
        endswith(f, ".md") || continue
        file = joinpath(dir, f)
        s = read(file, String)
        # Fences, then double-backtick spans (which may hold a backtick), then
        # single ones.
        prose = replace(s, r"```.*?```"s => "", r"``.+?``" => "", r"`[^`\n]*`" => "")
        for m in eachmatch(r"\]\(([^)\s]+)\)", prose)
            link = m[1]
            occursin(EXTERNAL, link) && continue
            path = first(split(link, '#'))
            if !isempty(path) && !isfile(normpath(joinpath(dir, path)))
                push!(bad, "$(relpath(file, SITE)): $link")
            end
        end
    end
    return bad
end

# A page this script writes or copies must also be one Quarto renders and the
# navbar lists, which docs/_quarto.yml says by hand: a note needs its name in the
# render list and in the menu, a reference page in the menu.
function check_config()
    yml = read(joinpath(DOCS, "_quarto.yml"), String)
    missing = [f for f in ["reference/index.qmd"; ["reference/$p.qmd" for (p, _) in PAGES]]
               if !occursin(f, yml)]
    append!(missing, filter(f -> count(f, yml) < 2, NOTES))
    isempty(missing) ||
        error("not in docs/_quarto.yml (render list and Notes or Reference menu):\n  " *
              join(missing, "\n  "))
end

function coverage(entries)
    for mod in (Honghu, Honghu.Problems)
        ex = filter(n -> n !== nameof(mod), names(mod))
        und = filter(n -> !Docs.hasdoc(mod, n), ex)
        println("coverage: $mod exports $(length(ex)) names, $(length(ex) - length(und)) documented",
                isempty(und) ? "" : "; undocumented: " * join(und, ", "))
    end
    println("coverage: $(count(e -> e.public, entries)) public and ",
            "$(count(e -> !e.public, entries)) internal entries on the site")
end

function main(args)
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    check_config()
    entries, moddocs = collect_entries()
    refmap = build_refmap(entries)
    write_reference(entries, moddocs, refmap, sha)
    write_index()
    coverage(entries)
    "--generate-only" in args && return
    render()
    write_twins_and_bundle(sha)
    bad = [check_html(); check_markdown()]
    isempty(bad) || error("$(length(bad)) broken links:\n  " * join(bad, "\n  "))
    println("site built in ", relpath(SITE, ROOT), ": every link and anchor resolves")
end

main(ARGS)
