#!/bin/bash
# Benchmark declearn across versions, using era-specific environments.
#
# Each declearn version is benchmarked in the venv that matches its
# original dependency era. This avoids issues like NumPy 2.0 breaking
# old declearn versions that used np.array(copy=False).

set -euo pipefail

cd "$(dirname "$0")"

# Map a declearn version to the venv that should benchmark it.
get_venv_for_version() {
    local version="$1"
    case "$version" in
        2.7.*|2.8.*)
            echo "$HOME/.venvs/declearn311"
            ;;
        2.4.*|2.5.*|2.6.*)
            echo "$HOME/.venvs/declearn311-mid"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Default versions if none specified
if [ $# -eq 0 ]; then
    VERSIONS=("2.4.0" "2.5.0" "2.6.0" "2.7.0" "2.8.0")
else
    VERSIONS=("$@")
fi

for VERSION in "${VERSIONS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Benchmarking declearn ${VERSION}"
    echo "=========================================="

    # Pick the right venv for this version
    VENV=$(get_venv_for_version "$VERSION")
    if [ -z "$VENV" ]; then
        echo "ERROR: No venv configured for declearn ${VERSION}, skipping"
        continue
    fi
    if [ ! -d "$VENV" ]; then
        echo "ERROR: Venv $VENV does not exist, skipping"
        continue
    fi

    echo "Using venv: $VENV"

    # Activate the chosen venv (subshell so it doesn't leak between iterations)
    (
        source "$VENV/bin/activate"

        # Try the version as-is, fall back to .post1 if needed
        if pip install "declearn==${VERSION}" --no-deps --quiet 2>/dev/null; then
            echo "Installed declearn==${VERSION}"
        elif pip install "declearn==${VERSION}.post1" --no-deps --quiet 2>/dev/null; then
            echo "Installed declearn==${VERSION}.post1 (post-release of ${VERSION})"
        else
            echo "ERROR: Could not install declearn==${VERSION} or any post-release"
            exit 1
        fi

        # Resolve the real git commit hash from the base version tag
        REAL_SHA=$(git -C declearn rev-parse "v${VERSION}^{commit}")
        echo "Resolved v${VERSION} to commit ${REAL_SHA:0:8}"

        asv run \
            --python=same \
            --set-commit-hash="$REAL_SHA" \
            -a repeat=3 \
            --record-samples \
            --show-stderr \
            || true
    ) || echo "WARNING: Version ${VERSION} sweep had a fatal error; continuing."
done

echo ""
echo "--- Generating HTML report ---"
asv publish

echo ""
echo "Done. To view:"
echo "  cd $(pwd)"
echo "  asv preview"