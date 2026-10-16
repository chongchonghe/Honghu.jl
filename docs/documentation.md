# How this documentation is built

This page explains how the documentation site of Honghu.jl is made, where each page comes from, and what to edit to change it. It is written for anyone who changes the code or the documentation, coding agents included.

## What the site contains

| Part | Source | How it gets onto the site |
|:---|:---|:---|
| Home page | `README.md` | copied to `docs/index.qmd` by `docs/make.jl`, with a few links rewritten |
| Reference | the docstrings in `src/` | written to `docs/reference/*.qmd` by `docs/make.jl` |
| Notes | `docs/*.md`: the four technical notes and this page | rendered as they are |
| Figures | `docs/figures/` | made by `docs/make_figures.jl` and `docs/make_animations.jl`, except the logo images `honghu-*.png`, which no script makes |

Every page is published twice, from the same source and in the same build: as HTML, and as Markdown. The Markdown copy of `x.html` is `x.md` in the same folder, and the line "Markdown: this page" at the top of each HTML page links to it. The copy is the page's source itself. The only change is that links to other pages point at their `.md` copies instead of their `.qmd` sources. Two more files sit at the site root: `honghu-docs.md`, which holds all pages in one file in navigation order, and `llms.txt`, an index of the Markdown pages in the llms.txt format. Both exist so that a person or a coding agent can read the whole documentation as plain text.

`docs/index.qmd` and everything in `docs/reference/` are build products. They are not in git, and the next build overwrites them, so edit the README and the docstrings instead.

## How a build runs

`julia --project=. docs/make.jl` does five things, in this order.

1. **Collect the docstrings.** It loads Honghu and reads every docstring in `Honghu`, `Honghu.Problems` and `Honghu.CGS` through `Docs.meta`. Each documented name becomes one entry, on the reference page its source file belongs to; the `PAGES` table at the top of `docs/make.jl` assigns files to pages. On each page, exported names come first, under "Exported", and documented names that are not exported come after, under "Internals". The constants in `CGS` count as public, because that module exports nothing by design. A name without a docstring does not appear at all.
2. **Write the pages.** Each docstring is copied as written, with five small changes (see [A docstring](#a-docstring) below). The script also writes `docs/reference/index.qmd`, which holds the `Honghu` module docstring and a list of all pages with their exported names, and `docs/index.qmd` from the README.
3. **Render.** `quarto render docs` turns the pages into HTML in `docs/_site/`. The Quarto project is `docs/_quarto.yml`. Its `render:` list says which files are pages, so the scripts and the internal notes elsewhere in `docs/` stay off the site. The Lua filter `docs/source-links.lua` adds the Markdown links to the top of every page, and makes the leading `# Title` of a note its page title.
4. **Write the Markdown copies,** the bundle `honghu-docs.md` and `llms.txt` into `docs/_site/`.
5. **Check,** and fail on any problem listed in [What fails the build](#what-fails-the-build).

After Honghu is precompiled, a full build takes about 15 seconds. It runs no simulation, and no code block in a docstring or a page is ever executed.

## Building and previewing locally

You need Julia, with the project's own environment (there is no separate docs environment), and Quarto 1.9 or newer on the `PATH`. CI uses Quarto 1.9.38.

```bash
julia --project=. docs/make.jl                  # full build into docs/_site
julia --project=. docs/make.jl --generate-only  # write the pages only, no render
quarto preview docs                             # live preview, after --generate-only
python3 -m http.server -d docs/_site            # serve a full build at http://localhost:8000
```

`quarto preview` updates the HTML when a README-derived page or a note is saved, but not when a docstring changes. The reference pages are written from the docstrings, so re-run `docs/make.jl --generate-only` after editing one. The Markdown copies, the bundle and the link check exist only in a full build. Run one before you commit.

## How to update

### A docstring

Edit it in `src/`. The next build picks it up and nothing else needs to change. Write docstrings as you would for Documenter:

- Start with the signature, indented by four spaces. It becomes a highlighted Julia code block.
- Link to another documented name with `` [`name`](@ref) ``. The forms `` [`Honghu.name`](@ref) `` and `` [`name(args)`](@ref) `` link to the same entry. The target must have a docstring: a link to an undocumented name fails the build. Fix that by writing the missing docstring, not by removing the brackets. The form `[text](@ref name)` is not supported and also fails.
- Headers such as `# Example` are moved down three levels, so that they sit below the entry's own heading.
- An admonition, `!!! warning "Title"` followed by a body indented by four spaces, becomes a Quarto callout. `note`, `tip`, `warning` and `danger` are recognised.
- A `jldoctest` block is shown as a plain `julia` block.
- Write formulas as plain text in code, as the existing docstrings do, for example `phi c dt [ ... ] <= 1`. No math is rendered inside docstrings. Documenter's double-backtick math is not converted and shows as code, and a `$` in a docstring is string interpolation in Julia.

### A new source file

If a new file in `src/` holds docstrings, add it to the `PAGES` table in `docs/make.jl`, under the page it belongs to. Until you do, the build fails with "no reference page for these files". An entry that ends in `/`, like `"problems/"`, takes every file in that directory.

### A new reference page

Add an entry `"name" => ("Title", ["file.jl", ...])` to `PAGES`, and add `reference/name.qmd` to the Reference menu in `docs/_quarto.yml`. The `render:` list already takes every file in `reference/`.

### The front page

Edit `README.md`. The build copies it to `docs/index.qmd`. It drops the first line, `# Honghu.jl`, which becomes the page title, and it rewrites the links that are written for GitHub's view of the repository: `docs/figures/...` becomes `figures/...`, in Markdown links and in HTML `src` attributes, and the links to `future.md` and `../Hydro.jl` become GitHub URLs (`write_index` in `docs/make.jl`). Any other relative link in the README must point at something on the site, or the link check fails. To link to another file in the repository, use its full GitHub URL.

### A note

The notes are plain Markdown files in `docs/` that also read well on GitHub. To add one:

1. Write `docs/name.md`. Begin it with `# Title`, which becomes the page title, and give it no YAML front matter. Use `$...$` and `$$...$$` for math. MathJax renders it on the site, and the Markdown copy keeps it exactly as written.
2. Add `name.md` to `NOTES` in `docs/make.jl`. This gives it a Markdown copy and a place in the bundle and in `llms.txt`.
3. Add `name.md` to the `render:` list and to the Notes menu in `docs/_quarto.yml`.

The build checks that every name in `NOTES` and every reference page also appears in `docs/_quarto.yml`. If you miss step 3, the build fails instead of leaving the page off the site.

To link from one note to another, use a relative link to its `.md` file, such as `[the findings](rsla-findings.md)`. Quarto turns it into a link to the HTML page, and the Markdown copy keeps it, where it points at the other copy. Images go in `docs/figures/` and are linked as `figures/name.png`. A link to a file that is not on the site, such as a script in `examples/`, must be a full GitHub URL.

### The navigation and the look

`docs/_quarto.yml` holds the navbar, the menus and the theme, and names the navbar icon and the favicon. It is written by hand and no script regenerates it.

## What fails the build

The build stops with an error, and writes nothing to deploy, in each of these cases.

| Message | Cause | Fix |
|:---|:---|:---|
| `no reference page for these files; add them to PAGES` | a file in `src/` has docstrings but no page | add it to `PAGES` in `docs/make.jl` |
| `@ref links with no docstring behind them` | a `` [`name`](@ref) `` whose `name` has no docstring, or an unsupported `@ref` form | write the docstring, or correct the name |
| `not in docs/_quarto.yml` | a note or reference page that `docs/_quarto.yml` does not list | add it to the `render:` list and the menu |
| `duplicate anchors` | two entries would get the same anchor | rename one of them, or refine `slug` in `docs/make.jl` |
| `quarto warned` | any warning from `quarto render` | read the warning; usually a link to a `.qmd` or `.md` that does not exist |
| `quarto render failed` | Quarto itself stopped | read Quarto's output above the error |
| `broken links` | a relative `href`, `src` or `#anchor` in the HTML, or a relative link in a Markdown copy, that does not resolve | fix the link in its source; the message names the page |

The anchor of an entry is its name, with `!` written as `-bang` and every other character that is not a letter, a digit, `_` or `-` replaced by `-`. So `run!` is `#run-bang` and `CGS.a_rad` is `#CGS-a_rad`.

## Deployment

`.github/workflows/docs.yml` builds the site on every push and every pull request, and deploys it to GitHub Pages from `main`, at <https://chongchonghe.github.io/Honghu.jl/>. A pull request that breaks the build fails its check before it is merged. Deployment needs one setting in the repository: Settings, Pages, Build and deployment, Source "GitHub Actions". The workflow pins Quarto to 1.9.38, the version the site was built and checked with. Raise it on purpose, after a local build with the new version, because Quarto's link handling has changed between releases.

## Why it is built this way

- **Quarto and a script in the repository, not Documenter or QuartoDocBuilder.jl.** The site needs HTML and Markdown from one source, and Documenter's Markdown backend is no longer maintained. QuartoDocBuilder.jl was examined and rejected. It is not registered, it leaves Documenter's `@ref` links broken (the docstrings have 258 of them), it cannot put several names on one page, and it rewrites `_quarto.yml` on every build. `docs/make.jl` is about 420 lines and does only what this package needs.
- **The Markdown copy is the source, published as `.md`.** GitHub Pages serves `.qmd` files as downloads and `.md` files as text. Publishing the source itself, rather than rendering a second output format, means the copy cannot drift from the HTML, and the math stays exactly as written.
- **No Quarto `llms-txt`.** In Quarto 1.9 the Markdown it produces garbles math, and the fix is only in the 1.11 pre-releases. `docs/make.jl` writes `llms.txt` itself. A Quarto release with the fix could take over that file, but the bundle and the per-page links would still be needed.
- **A link check of its own.** Quarto warns about some missing links and has no strict mode, so `docs/make.jl` checks every link and anchor in the rendered site.
- **`data-noresolveinput` on the Markdown links.** Quarto resolves every link in a page against the project's inputs, raw HTML included. It would warn about the `.md` copies, which do not exist yet while it renders. The undocumented attribute `data-noresolveinput` turns that off for the three links the filter adds.
