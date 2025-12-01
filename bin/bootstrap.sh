#!/usr/bin/env bash
set -Eeuo pipefail

trap 'err "Failed at line $LINENO: $BASH_COMMAND"' ERR

usage() {
  cat <<'USAGE'
Bootstrap the whaddya-want development environment (Python 3.11 enforced).

Usage: bootstrap.sh [options]

Options:
  --mode <venv|conda>   Choose Python env manager (default: venv).
  --venv-path <path>    Path for the virtual environment (default: .venv).
  --conda-env <name>    Conda env name (default: aai).
  --force               Recreate the requested environment from scratch.
  --skip-system-check   Skip verification of system dependencies.
  --region <name>       AWS region (required unless AWS_REGION is already set).
  -h, --help            Show this help and exit.

Notes:
  • venv path uses a real Python 3.11 interpreter (not 3.12/3.13).
  • conda path creates env with python=3.11, then pip-installs requirements.
USAGE
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "This script must be executed directly, not sourced." >&2
  return 1
fi

require_aws_profile() {
  if [[ -z "${AWS_PROFILE:-}" ]]; then
    err "AWS_PROFILE is not set."
  fi
}

COMMON_REGION_ARG=""
COMMON_REMAINING_ARGS=()
parse_region_and_remainder() {
  COMMON_REGION_ARG=""
  COMMON_REMAINING_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --region)
        COMMON_REGION_ARG="${2:-}"
        shift 2
        ;;
      --region=*)
        COMMON_REGION_ARG="${1#*=}"
        shift
        ;;
      *)
        COMMON_REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

apply_region() {
  local provided="$1"
  if [[ -n "$provided" ]]; then
    export AWS_REGION="$provided"
  elif [[ -n "${AWS_REGION:-}" ]]; then
    export AWS_REGION
  else
    err "you must set --region or AWS_REGION."
  fi
  export DEFAULT_AWS_REGION="$AWS_REGION"
  export AWS_DEFAULT_REGION="$AWS_REGION"
}

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

relative_path() {
  python3 - "$1" "$2" <<'PY'
import os,sys
print(os.path.relpath(sys.argv[1], sys.argv[2]))
PY
}

# --- find/enforce Python 3.11 ---
PY311_BIN=""

is_py311() {
  "$1" - <<'PY' 2>/dev/null || exit 1
import sys
print(int(sys.version_info[:2] == (3,11)))
PY
}

find_python311() {
  # 1) Respect $PYTHON if provided and is 3.11
  if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    if [[ "$(is_py311 "$PYTHON")" == "1" ]]; then PY311_BIN="$(command -v "$PYTHON")"; return 0; fi
    warn "\$PYTHON points to $( "$PYTHON" -V 2>&1 ); need Python 3.11"
  fi

  # 2) Direct python3.11 on PATH
  if command -v python3.11 >/dev/null 2>&1 && [[ "$(is_py311 "$(command -v python3.11)")" == "1" ]]; then
    PY311_BIN="$(command -v python3.11)"; return 0
  fi

  # 3) If python3 is 3.11, use it
  if command -v python3 >/dev/null 2>&1 && [[ "$(is_py311 "$(command -v python3)")" == "1" ]]; then
    PY311_BIN="$(command -v python3)"; return 0
  fi

  # 4) Try pyenv auto-install/use
  if command -v pyenv >/dev/null 2>&1; then
    local ver="3.11.9"
    if ! pyenv versions --bare | grep -qx "$ver"; then
      log "Installing Python $ver via pyenv (first time only)..."
      pyenv install -s "$ver"
    fi
    local root; root="$(pyenv root)"
    local cand="$root/versions/$ver/bin/python3.11"
    if [[ -x "$cand" ]] && [[ "$(is_py311 "$cand")" == "1" ]]; then
      PY311_BIN="$cand"; return 0
    fi
  fi

  # 5) Friendly failure with remedies
  err $'Could not find a Python 3.11 interpreter.\n\
Hints:\n  • macOS (Homebrew):   brew install python@3.11\n  • pyenv:               brew install pyenv && pyenv install 3.11.9\n  • Ubuntu/Debian:       apt-get install python3.11 python3.11-venv'
}

SCRIPT_DIR=$PWD
REPO_ROOT=$PWD
MODE="venv"
VENV_PATH="${REPO_ROOT}/.venv"
CONDA_ENV_NAME="aai"
FORCE=0
SKIP_SYSTEM_CHECK=0

parse_region_and_remainder "$@"
set -- "${COMMON_REMAINING_ARGS[@]}"
require_aws_profile
apply_region "$COMMON_REGION_ARG"

# --- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE=$(lower "${2:-}"); shift ;;
    --mode=*)      MODE=$(lower "${1#*=}") ;;
    --venv-path)   VENV_PATH="${2:-}"; shift ;;
    --venv-path=*) VENV_PATH="${1#*=}" ;;
    --conda-env)   CONDA_ENV_NAME="${2:-}"; shift ;;
    --conda-env=*) CONDA_ENV_NAME="${1#*=}" ;;
    --force)       FORCE=1 ;;
    --skip-system-check) SKIP_SYSTEM_CHECK=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) err "Unknown option: $1" ;;
  esac
  shift || true
done

[[ "$MODE" = "venv" || "$MODE" = "conda" ]] || err "Unsupported mode '$MODE'. Use 'venv' or 'conda'."

ensure_cmd() { command -v "$1" >/dev/null 2>&1 || err "Required command '$1' not found in PATH."; }
need_file() { [[ -f "$1" ]] || err "Missing required file: $1"; }

check_system_dependencies() {
  local miss=()
  for c in aws; do command -v "$c" >/dev/null 2>&1 || miss+=("$c"); done
  if [[ ${#miss[@]} -gt 0 ]]; then
    warn "Missing required commands: ${miss[*]}"
    warn "Install them (e.g., Homebrew: brew install awscli) then re-run."
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
  need_file "${REPO_ROOT}/client/requirements.txt"
  need_file "${REPO_ROOT}/lambda/broker/requirements.txt"

  local env_exists=0
  if conda info --envs | awk 'NF{print $1}' | sed 's/*//' | grep -Fxq "$CONDA_ENV_NAME"; then
    env_exists=1
  fi

  if [[ $FORCE -eq 1 && $env_exists -eq 1 ]]; then
    log "Removing existing Conda env '$CONDA_ENV_NAME'."
    conda env remove --name "$CONDA_ENV_NAME" || true
    env_exists=0
  fi

  if [[ $env_exists -eq 0 ]]; then
    log "Creating Conda env '$CONDA_ENV_NAME' with python=3.11."
    conda create -y -n "$CONDA_ENV_NAME" python=3.11
  else
    # Verify Python version
    local v
    v="$(conda run --no-capture-output --name "$CONDA_ENV_NAME" python - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    if [[ "$v" != "3.11" ]]; then
      err "Conda env '$CONDA_ENV_NAME' is Python $v; use --force to recreate with Python 3.11."
    fi
    log "Using existing Conda env '$CONDA_ENV_NAME' (Python $v)."
  fi

  local pip_cmd=(conda run --no-capture-output --name "$CONDA_ENV_NAME" python -m pip)
  "${pip_cmd[@]}" install --upgrade pip wheel setuptools
  for req in "${REPO_ROOT}/client/requirements.txt" "${REPO_ROOT}/lambda/broker/requirements.txt"; do
    local rel; rel=$(relative_path "$req" "${REPO_ROOT}")
    need_file "$req"
    log "Installing Python requirements from ${rel} (Conda env)."
    "${pip_cmd[@]}" install -r "$req"
  done

  log "Conda environment ready. Activate with:"
  echo "  conda activate $CONDA_ENV_NAME"
}

ensure_venv_environment() {
  need_file "${REPO_ROOT}/client/requirements.txt"
  need_file "${REPO_ROOT}/lambda/broker/requirements.txt"

  find_python311
  log "Using Python 3.11 at: $PY311_BIN"

  if [[ $FORCE -eq 1 && -d "$VENV_PATH" ]]; then
    log "Removing existing venv at '$VENV_PATH'."
    rm -rf "$VENV_PATH"
  fi

  if [[ ! -d "$VENV_PATH" ]]; then
    log "Creating venv at '$VENV_PATH' with Python 3.11."
    "$PY311_BIN" -m venv "$VENV_PATH"
  else
    log "Reusing venv at '$VENV_PATH'."
  fi

  local py="${VENV_PATH}/bin/python"
  [[ -x "$py" ]] || err "Virtual env at '$VENV_PATH' is missing Python."

  # Verify it's 3.11
  local v
  v="$("$py" - <<'PY'
import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  [[ "$v" == "3.11" ]] || err "Venv Python is $v; remove venv or pass --force to recreate with Python 3.11."

  log "Upgrading pip, wheel, setuptools."
  "$py" -m pip install --upgrade pip wheel setuptools

  for req in "${REPO_ROOT}/client/requirements.txt" "${REPO_ROOT}/lambda/broker/requirements.txt"; do
    local rel; rel=$(relative_path "$req" "${REPO_ROOT}")
    need_file "$req"
    log "Installing requirements from ${rel} into venv."
    "$py" -m pip install -r "$req"
  done

  local act; act=$(relative_path "${VENV_PATH}/bin/activate" "${REPO_ROOT}")
  log "venv ready. Activate with:"
  echo "  source ${act}"
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
