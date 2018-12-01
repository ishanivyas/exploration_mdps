# Run this like: source path.sh
THIS_SCRIPT_DIR="$(dirname ${BASH_SOURCE[@]})"
if [[ "${THIS_SCRIPT_DIR}" == "." ]]; then
    THIS_SCRIPT_DIR="$(pwd)"
elif [[ ${THIS_SCRIPT_DIR:0:1} != / ]]; then
    THIS_SCRIPT_DIR="$(pwd)/$THIS_SCRIPT_DIR"
fi
export PYTHONPATH=.:$THIS_SCRIPT_DIR
