# Releasing Honghu.jl

A release is a published GitHub release with a tag `vX.Y.Z`. Zenodo archives every published release of the repository automatically, gives it a DOI, and takes the record's metadata (title, author, description, keywords, license, version) from `CITATION.cff`. There is no `.zenodo.json`; Zenodo would use it instead of `CITATION.cff` if there were, and it is needed only for funding, Zenodo communities or related identifiers.

## One-time setup

These steps need the maintainer's own accounts.

1. **Link Zenodo to GitHub.** Log in at <https://zenodo.org> with GitHub, or link GitHub to an existing Zenodo account under *Profile → Linked accounts*.
2. **Enable the repository.** At <https://zenodo.org/account/settings/github/>, find `chongchonghe/Honghu.jl` and turn its switch on. Zenodo archives only releases published *after* this; a release published before it is not picked up.
3. **Add the ORCID.** Add `orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"` to the author in `CITATION.cff`, and link the ORCID to the Zenodo profile as well, so that the record is attributed.

## Each release

1. **Choose the version**, following [Semantic Versioning](https://semver.org/). Before 1.0, a breaking change raises the minor version (`0.1.0` to `0.2.0`), anything else the patch version.
2. **Write it in two places**: `version` in `Project.toml` and `version` in `CITATION.cff`. `test/test_metadata.jl` fails if they differ, or if `CHANGELOG.md` has no section for the version.
3. **Date the changelog**: in `CHANGELOG.md`, replace `Unreleased` in the version's heading by the release date, `YYYY-MM-DD`, and start a new `## [Unreleased]` section above it for what comes next.
4. **Check**:

   ```bash
   uvx --from cffconvert cffconvert --validate -i CITATION.cff   # the metadata Zenodo will read
   uvx --from cffconvert cffconvert -f zenodo -i CITATION.cff    # ... and what Zenodo will make of it
   julia --project=. -t 4 test/runtests.jl
   julia --project=. docs/make.jl
   ```

5. **Commit and push** to `main`, and wait for both CI workflows (Tests, Docs) to pass.
6. **Publish the release**, with the changelog section as its notes:

   ```bash
   v=0.1.0
   awk -v v="$v" '$0 ~ "^## \\[" v "\\]" {f=1; next} /^## \[/ {f=0} f' CHANGELOG.md > /tmp/notes-$v.md
   gh release create v$v --target main --title "Honghu.jl v$v" --notes-file /tmp/notes-$v.md
   ```

   Do not mark it as a draft: Zenodo archives published releases.
7. **Check the record.** Within a few minutes the release appears at <https://zenodo.org/account/settings/github/repository/chongchonghe/Honghu.jl> with a DOI. If it failed, the *Errors* panel there says why; it is usually the metadata. Fix the metadata, then publish a new patch release.

## After the first release

Zenodo issues two DOIs: one for the version just archived, and a *concept DOI* that always resolves to the latest version. Cite the concept DOI unless a specific version matters.

Done after v0.1.0 (2026-10-23), whose record is <https://zenodo.org/records/23569082> (version DOI `10.5281/zenodo.23569082`): the concept DOI, `10.5281/zenodo.23569081`, is in `CITATION.cff` as `doi` and in a badge at the top of `README.md`. Zenodo ignores `doi` when it reads `CITATION.cff` (`cffconvert -f zenodo` drops it), so it does not clash with the new DOI each release gets. Nothing needs to change for later releases.

## If Honghu is registered in Julia's General registry

Registration is a separate step (comment `@JuliaRegistrator register` on a commit). Once the package is registered, a version becomes public through the registry, and the TagBot GitHub Action creates the GitHub release and tag for it, which Zenodo then archives as above. Steps 1 to 5 still apply; step 6 becomes the registration comment. Neither JuliaRegistrator's app nor TagBot is set up yet.
