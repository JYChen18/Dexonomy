# wrap mesh_viewer.py in a function to make it easier to call
# e.g., from command line or other scripts
# usage: ./viewer.sh <> <> <>...

function mesh_view {
  python3 script/mesh_viewer_colored.py "$@"
}

mesh_view "$@"