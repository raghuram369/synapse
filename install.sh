#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${SYNAPSE_PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "❌ Python 3 is required (python3 or python)."
    exit 1
  fi
fi

NO_ONBOARD=0
NON_INTERACTIVE=0
RUN_NODE_BOOTSTRAP=0
DB_PATH="${SYNAPSE_DB_PATH:-}"

usage() {
  cat <<'USAGE'
Usage: install.sh [options]

Runtime bootstrap helpers for Synapse.

Options:
  --no-onboard           Skip interactive/auto onboarding step
  --non-interactive      Run in unattended mode (no prompts)
  --with-node-bootstrap   Run npx @synapse-memory/setup@latest after install
  --db <path>            Pass --db to onboarding
  -h, --help             Show this help
USAGE
}

while (($# > 0)); do
  case "$1" in
    --no-onboard)
      NO_ONBOARD=1
      ;;
    --non-interactive)
      NON_INTERACTIVE=1
      ;;
    --with-node-bootstrap)
      RUN_NODE_BOOTSTRAP=1
      ;;
    --db)
      shift
      if (($# < 1)); then
        echo "--db requires a path argument"
        exit 1
      fi
      DB_PATH="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown flag: $1"
      usage
      exit 1
      ;;
  esac
  shift
 done

echo "🧠 Synapse bootstrap: ensuring package install..."
if ! "$PYTHON_BIN" -m pip install --upgrade --user synapse-ai-memory; then
  echo "❌ Failed to install synapse-ai-memory"
  exit 1
fi

echo "🧠 Synapse bootstrap: preparing managed runtime"
"$PYTHON_BIN" - <<'PY'
import installer
print(installer._ensure_mcp_wrapper())
print(f"runtime={installer._runtime_root()}")
print(f"wrapper={installer._runtime_python_path()}")
PY

if command -v synapse >/dev/null 2>&1; then
  SYNAPSE_CMD="synapse"
else
  SYNAPSE_CMD="$PYTHON_BIN -m cli"
fi

echo "🧠 Synapse bootstrap: running doctor check"
if ! $SYNAPSE_CMD doctor --non-interactive --json $(if [[ -n "$DB_PATH" ]]; then echo "--db" "$DB_PATH"; fi); then
  echo "⚠️  Doctor reported errors. Please review output above and re-run onboarding if needed."
fi

if (( NO_ONBOARD == 0 )); then
  echo "🧠 Synapse bootstrap: running onboard"
  ONBOARD_ARGS=( )
  if (( NON_INTERACTIVE == 1 )); then
    ONBOARD_ARGS+=(--non-interactive --flow quickstart)
  fi
  if [[ -n "$DB_PATH" ]]; then
    ONBOARD_ARGS+=(--db "$DB_PATH")
  fi

  set +e
  $SYNAPSE_CMD onboard "${ONBOARD_ARGS[@]}"
  onboard_rc=$?
  set -e
  if (( onboard_rc != 0 )); then
    echo "⚠️  Onboarding returned status ${onboard_rc}. You can re-run: synapse onboard"
  fi
else
  echo "🧠 Synapse bootstrap: skipping onboard (--no-onboard)"
fi

if (( RUN_NODE_BOOTSTRAP == 1 )); then
  if command -v npx >/dev/null 2>&1; then
    echo "🧠 Synapse bootstrap: running optional Node setup"
    if ! npx @synapse-memory/setup@latest; then
      echo "⚠️  Node setup did not complete. Continuing with Python path only."
    fi
  else
    echo "⚠️  npx not found; skipping optional node bootstrap."
  fi
fi

echo "✅ Synapse installed successfully."
echo "Runtime: managed (isolated)"
echo "Next: synapse onboard"
