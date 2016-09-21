#! /bin/bash

set -e

mumax3 -vet *.mx3

mumax3 -paranoid=false -failfast -cache /tmp -f -http "" *.go *.mx3

