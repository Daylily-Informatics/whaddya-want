#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Bootstrap the whaddya-want development environment.

Usage: bootstrap.sh [options]

Options:
  --mode <venv|conda>   Choose how to manage the Python environment (default: venv).
  --venv-path <path>    Location for the Python virtual environment (default: .venv).
  --conda-env <name>    Name of the Conda environment to create or update (default: aai).
  --force               Recreate the requested environment from scratch.
  --skip-system-check   Skip verification of optional system dependencies.
  -h, --help            Show this help message and exit.
USAGE
}

log() {
    printf '\033[1;34m==>\033[0m %s\n' "$*"
}

warn() {
    printf '\033[1;33m[WARN]\033[0m %s\n' "$*"
}

err() {
    printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2
}

relative_path() {
    python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"
}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE="venv"
VENV_PATH="${REPO_ROOT}/.venv"
CONDA_ENV_NAME="aai"
FORCE=0
SKIP_SYSTEM_CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE=${2,,}
            shift
            ;;
        --mode=*)
            MODE=${1#*=}
            MODE=${MODE,,}
            ;;
        --venv-path)
            VENV_PATH=$2
            shift
            ;;
        --venv-path=*)
            VENV_PATH=${1#*=}
            ;;
        --conda-env)
            CONDA_ENV_NAME=$2
            shift
            ;;
        --conda-env=*)
            CONDA_ENV_NAME=${1#*=}
            ;;
        --force)
            FORCE=1
            ;;
        --skip-system-check)
            SKIP_SYSTEM_CHECK=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ "${MODE}" != "venv" && "${MODE}" != "conda" ]]; then
    err "Unsupported mode '${MODE}'. Use 'venv' or 'conda'."
    exit 1
fi

ensure_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Required command '$1' not found in PATH."
        return 1
    fi
}

check_system_dependencies() {
    local missing=()
    for cmd in python3 aws; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        warn "The following required commands are missing: ${missing[*]}"
        warn "Install them using your system package manager before continuing."
        exit 1
    fi

    local optional=(ffmpeg)
    for cmd in "${optional[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            warn "Optional dependency '$cmd' was not detected. Audio capture or playback features may be limited."
        fi
    done
}

ensure_conda_environment() {
    ensure_command conda

    local env_exists=0
    if conda info --envs | awk '{print $1}' | sed 's/*//' | grep -Fxq "${CONDA_ENV_NAME}"; then
        env_exists=1
    fi

    if [[ $FORCE -eq 1 && $env_exists -eq 1 ]]; then
        log "Removing existing Conda environment '${CONDA_ENV_NAME}'."
        conda env remove --name "${CONDA_ENV_NAME}"
        env_exists=0
    fi

    if [[ $env_exists -eq 0 ]]; then
        log "Creating Conda environment '${CONDA_ENV_NAME}'."
        conda env create --name "${CONDA_ENV_NAME}" --file "${REPO_ROOT}/conf/aai.yaml"
    else
        log "Updating Conda environment '${CONDA_ENV_NAME}'."
        conda env update --name "${CONDA_ENV_NAME}" --file "${REPO_ROOT}/conf/aai.yaml" --prune
    fi

    local pip_cmd=(conda run --no-capture-output --name "${CONDA_ENV_NAME}" python -m pip)
    "${pip_cmd[@]}" install --upgrade pip
    for req in "${REPO_ROOT}/client/requirements.txt" "${REPO_ROOT}/lambda/broker/requirements.txt"; do
        local rel
        rel=$(relative_path "$req" "${REPO_ROOT}")
        log "Installing Python requirements from ${rel} in Conda env."
        "${pip_cmd[@]}" install -r "$req"
    done

    log "Conda environment '${CONDA_ENV_NAME}' is ready. Activate it with:"
    echo "  conda activate ${CONDA_ENV_NAME}"
}

ensure_venv_environment() {
    ensure_command python3

    if [[ $FORCE -eq 1 && -d "${VENV_PATH}" ]]; then
        log "Removing existing virtual environment at '${VENV_PATH}'."
        rm -rf "${VENV_PATH}"
    fi

    if [[ ! -d "${VENV_PATH}" ]]; then
        log "Creating virtual environment at '${VENV_PATH}'."
        python3 -m venv "${VENV_PATH}"
    else
        log "Reusing virtual environment at '${VENV_PATH}'."
    fi

    local python_bin="${VENV_PATH}/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        err "Virtual environment at '${VENV_PATH}' is missing the Python executable."
        exit 1
    fi

    log "Upgrading pip in the virtual environment."
    "${python_bin}" -m pip install --upgrade pip

    for req in "${REPO_ROOT}/client/requirements.txt" "${REPO_ROOT}/lambda/broker/requirements.txt"; do
        local rel
        rel=$(relative_path "$req" "${REPO_ROOT}")
        log "Installing Python requirements from ${rel} into the virtual environment."
        "${python_bin}" -m pip install -r "$req"
    done

    local activate_path
    activate_path=$(relative_path "${VENV_PATH}/bin/activate" "${REPO_ROOT}")
    log "Virtual environment ready. Activate it with:"
    echo "  source ${activate_path}"
    echo "Don't forget to export PYTHONPATH='${REPO_ROOT}/src' when running local tools."
}

if [[ ${SKIP_SYSTEM_CHECK} -eq 0 ]]; then
    check_system_dependencies
else
    log "Skipping system dependency checks as requested."
fi

case "${MODE}" in
    conda)
        ensure_conda_environment
        ;;
    venv)
        ensure_venv_environment
        ;;
esac

log "Bootstrap complete."
