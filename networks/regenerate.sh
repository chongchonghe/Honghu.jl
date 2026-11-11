#!/usr/bin/env bash
# Regenerate Honghu's chemical networks from their JAFF network files.

set -euo pipefail

# Pinned so that the generated code is the same on every machine.
SYMPY_VERSION=1.14.0
TEMPLATE=honghu
NETWORKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$NETWORKS_DIR")"

# network folder -> generated file, relative to the repository
TARGETS=(
    "hydrogen_iliev1:src/networks/hydrogen_iliev1.jl"
    "hydrogen_iliev2:src/networks/hydrogen_iliev2.jl"
    "hydrogen_photo_only:test/networks/hydrogen_photo_only.jl"
)

usage() {
    cat <<EOF
Usage: $(basename "$0") --jaff-path DIR [-h]

For every network folder in networks/ (its .jet, .jet.jfunc and jaffgen.toml), run
JAFF's jaffgen with the '$TEMPLATE' template and write the Julia file that Honghu
includes. Steps:
  1. make a throwaway Python environment and install JAFF from DIR into it,
     with sympy $SYMPY_VERSION;
  2. run jaffgen once per network;
  3. copy each generated network.jl to its place in the repository.

Options:
  --jaff-path DIR  a JAFF checkout that has the '$TEMPLATE' template
  -h, --help       show this message and exit
EOF
}

JAFF_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --jaff-path)
            [[ $# -ge 2 ]] || { echo "error: --jaff-path needs a directory" >&2; exit 1; }
            JAFF_PATH="$2"; shift 2 ;;
        *) echo "error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
    esac
done
[[ -n "$JAFF_PATH" ]] || { echo "error: --jaff-path is required" >&2; usage >&2; exit 1; }
[[ -d "$JAFF_PATH/src/jaff/templates/generator/$TEMPLATE" ]] ||
    { echo "error: '$JAFF_PATH' has no '$TEMPLATE' template" >&2; exit 1; }

export PIP_DISABLE_PIP_VERSION_CHECK=1
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Installing JAFF from $JAFF_PATH"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet -e "$JAFF_PATH"
"$WORK/venv/bin/pip" install --quiet "sympy==$SYMPY_VERSION"

for entry in "${TARGETS[@]}"; do
    name="${entry%%:*}"
    target="$REPO_DIR/${entry#*:}"
    echo "==> $name -> ${entry#*:}"
    "$WORK/venv/bin/jaffgen" \
        --config "$NETWORKS_DIR/$name/jaffgen.toml" \
        --network "$NETWORKS_DIR/$name/$name.jet" \
        --template "$TEMPLATE" \
        --outdir "$WORK/$name" > "$WORK/$name.log" 2>&1 ||
        { cat "$WORK/$name.log" >&2; exit 1; }
    mkdir -p "$(dirname "$target")"
    cp "$WORK/$name/network.jl" "$target"
done
echo "==> Done"
