#! /bin/bash

export MUMAX="$GOPATH/bin/mumax3"
echo "Using the mumax3 executable at: ${MUMAX}"

set -e

$MUMAX -vet *.mx3

$MUMAX -paranoid=false -failfast -cache /tmp -f -http "" *.go *.mx3

