-- Link every page to its Markdown twin and to the bundle of all pages, which
-- docs/make.jl writes after the render: `<page>.md` beside `<page>.html`, and
-- `honghu-docs.md` and `llms.txt` at the site root. Quarto resolves every
-- `<a href>` against the project's inputs and warns about a `.md` it cannot find,
-- which these are not; `data-noresolveinput` tells it to leave them alone.

function Pandoc(doc)
  -- The notes are plain Markdown that also reads well on GitHub, so they carry
  -- their title as a leading `# ...` rather than in front matter; make it the
  -- page title, as front matter would.
  local first = doc.blocks[1]
  if doc.meta.title == nil and first and first.t == "Header" and first.level == 1 then
    doc.meta.title = first.content
    doc.meta.pagetitle = pandoc.utils.stringify(first.content)
    table.remove(doc.blocks, 1)
  end
  local stem = pandoc.path.split_extension(pandoc.path.filename(quarto.doc.input_file))
  local twin = stem .. ".md"
  local root = quarto.project.offset
  quarto.doc.include_text("in-header",
    '<link rel="alternate" type="text/markdown" href="' .. twin .. '">')
  local line = '<p class="markdown-links">Markdown: <a data-noresolveinput href="' .. twin .. '">this page</a>'
    .. ' · <a data-noresolveinput href="' .. root .. '/honghu-docs.md">all pages in one file</a>'
    .. ' · <a data-noresolveinput href="' .. root .. '/llms.txt">llms.txt</a></p>'
  table.insert(doc.blocks, 1, pandoc.RawBlock("html", line))
  return doc
end
