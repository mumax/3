#! /bin/bash

# Optional arguments.
# Example usage: ./run.bash "/home/me/mumax/mumax3"
MUMAX="${1:-$GOPATH/bin/mumax3}"
echo "Using the mumax3 executable at: ${MUMAX}"

set -e

$MUMAX -vet *.mx3

$MUMAX -paranoid=false -failfast -cache /tmp -http "" -f *.go *.mx3
