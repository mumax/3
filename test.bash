#! /bin/bash

source ./make.bash || exit 1

go test -i github.com/mumax/3/... || exit 1
go test $PKGS  github.com/mumax/3/... || exit 1

(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
