#!/usr/bin/env bash
set -Eeuo pipefail

trap 'err "Failed at line $LINENO: $BASH_COMMAND"' ERR

usage() {
  cat <<'USAGE'
Bootstrap the whaddya-want development environment.

Usage: bootstrap.sh [options]

Options:
  --mode <venv|conda>   Choose Python env manager (default: venv).
  --venv-path <path>    Path for the virtual environment (default: .venv).
  --conda-env <name>    Conda env name (default: aai).
  --force               Recreate the requested environment from scratch.
  --skip-system-check   Skip verification of system dependencies.
  -h, --help            Show this help and exit.
USAGE
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; exit 1; }

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

relative_path() {
  python3 - "$1" "$2" <<'PY'
import os,sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
}

SCRIPT_DIR=$PWD
REPO_ROOT=$PWD
MODE="venv"
VENV_PATH="${REPO_ROOT}/.venv"
CONDA_ENV_NAME="aai"
FORCE=0
SKIP_SYSTEM_CHECK=0

# --- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE=$(lower "${2:-}")
      shift
      ;;
    --mode=*)
      MODE=$(lower "${1#*=}")
      ;;
    --venv-path)
      VENV_PATH="${2:-}"; shift ;;
    --venv-path=*)
      VENV_PATH="${1#*=}" ;;
    --conda-env)
      CONDA_ENV_NAME="${2:-}"; shift ;;
    --conda-env=*)
      CONDA_ENV_NAME="${1#*=}" ;;
    --force) FORCE=1 ;;
    --skip-system-check) SKIP_SYSTEM_CHECK=1 ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown option: $1" ;;
  esac
  shift || true
done

[[ "$MODE" = "venv" || "$MODE" = "conda" ]] || err "Unsupported mode '$MODE'. Use 'venv' or 'conda'."

ensure_cmd() { command -v "$1" >/dev/null 2>&1 || err "Required command '$1' not found in PATH."; }

need_file() { [[ -f "$1" ]] || err "Missing required file: $1"; }

check_system_dependencies() {
  local miss=()
  for c in python3 aws; do command -v "$c" >/dev/null 2>&1 || miss+=("$c"); done
  if [[ ${#miss[@]} -gt 0 ]]; then
    warn "Missing required commands: ${miss[*]}"
    warn "Install them (e.g., Homebrew: brew install awscli python) then re-run."
    exit 1
  fi

  # Optional but recommended
  for c in ffmpeg; do
    command -v "$c" >/dev/null 2>&1 || warn "Optional dep '$c' not found (media features may be limited)."
  done

  # AWS sanity (credentials/region)
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    err "AWS CLI is installed but not authenticated. Run 'aws configure sso' or 'aws configure'."
  fi
}

ensure_conda_environment() {
  ensure_cmd conda
  # Files must exist
  need_file "${REPO_ROOT}/conf/aai.yaml"
  need_file "${REPO_ROOT}/client/requirements.txt"
  need_file "${REPO_ROOT}/lambda/broker/requirements.txt"

  local env_exists=0
  # robust parse: read first column that isn't '#'
  if conda info --envs | awk 'NF{print $1}' | sed 's/*//' | grep -Fxq "$CONDA_ENV_NAME"; then
    env_exists=1
  fi

  if [[ $FORCE -eq 1 && $env_exists -eq 1 ]]; then
    log "Removing existing Conda env '$CONDA_ENV_NAME'."
    conda env remove --name "$CONDA_ENV_NAME" || true
    env_exists=0
  fi

  # If YAML has a name, prefer it; otherwise use --name
  if [[ $env_exists -eq 0 ]]; then
    if grep -qiE '^name:\s*' "${REPO_ROOT}/conf/aai.yaml"; then
      log "Creating Conda env from YAML name (conf/aai.yaml)."
      conda env create --file "${REPO_ROOT}/conf/aai.yaml"
    else
      log "Creating Conda env '$CONDA_ENV_NAME' from conf/aai.yaml."
      conda env create --name "$CONDA_ENV_NAME" --file "${REPO_ROOT}/conf/aai.yaml"
    fi
  else
    log "Updating Conda env '$CONDA_ENV_NAME'."
    #conda env update --name "$CONDA_ENV_NAME" --file "${REPO_ROOT}/conf/aai.yaml" --prune
  fi

  # Use the resolved env name (handles YAML-driven name)
  local use_env="$CONDA_ENV_NAME"
  if conda info --envs | awk 'NF{print $1}' | sed 's/*//' | grep -Fq "$(basename "$REPO_ROOT")"; then :; fi

  local pip_cmd=(conda run --no-capture-output --name "$use_env" python -m pip)
  "${pip_cmd[@]}" install --upgrade pip wheel
  for req in "${REPO_ROOT}/client/requirements.txt" "${REPO_ROOT}/lambda/broker/requirements.txt"; do
    local rel; rel=$(relative_path "$req" "${REPO_ROOT}")
    need_file "$req"
    log "Installing Python requirements from ${rel} (Conda env)."
    "${pip_cmd[@]}" install -r "$req"
  done

  log "Conda environment ready. Activate with:"
  echo "  conda activate $use_env"
}

ensure_venv_environment() {
  ensure_cmd python3
  need_file "${REPO_ROOT}/client/requirements.txt"
  need_file "${REPO_ROOT}/lambda/broker/requirements.txt"

  if [[ $FORCE -eq 1 && -d "$VENV_PATH" ]]; then
    log "Removing existing venv at '$VENV_PATH'."
    rm -rf "$VENV_PATH"
  fi

  if [[ ! -d "$VENV_PATH" ]]; then
    log "Creating venv at '$VENV_PATH'."
    python3 -m venv "$VENV_PATH"
  else
    log "Reusing venv at '$VENV_PATH'."
  fi

  local py="${VENV_PATH}/bin/python"
  [[ -x "$py" ]] || err "Virtual env at '$VENV_PATH' is missing Python."

  log "Upgrading pip & wheel."
  "$py" -m pip install --upgrade pip wheel

  for req in "${REPO_ROOT}/client/requirements.txt" "${REPO_ROOT}/lambda/broker/requirements.txt"; do
    local rel; rel=$(relative_path "$req" "${REPO_ROOT}")
    need_file "$req"
    log "Installing requirements from ${rel} into venv."
    "$py" -m pip install -r "$req"
  done

  local act; act=$(relative_path "${VENV_PATH}/bin/activate" "${REPO_ROOT}")
  log "venv ready. Activate with:"
  echo "  source ${act}"
  echo "Tip: add a src/pyproject and install it in editable mode to avoid PYTHONPATH exports."
}

if [[ $SKIP_SYSTEM_CHECK -eq 0 ]]; then
  check_system_dependencies
else
  log "Skipping system dependency checks as requested."
fi

case "$MODE" in
  conda) ensure_conda_environment ;;
  venv)  ensure_venv_environment  ;;
esac

log "Bootstrap complete."
