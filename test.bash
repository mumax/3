#! /bin/bash

source ./make.bash || exit 1

# The go test tool runs a vet check by default since go1.10.
# This vet check yields false negatives, so let's turn this off.
VETTESTFLAG=""
GOMINORVERSION=$( go version | sed -r 's/.*go1.([0-9]*).*/\1/g' )
if (( $GOMINORVERSION >= 10 )); then
    VETTESTFLAG="-vet=off"
fi

go test $VETTESTFLAG -i github.com/mumax/3/... || exit 1
go test $VETTESTFLAG $PKGS  github.com/mumax/3/... || exit 1

(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
