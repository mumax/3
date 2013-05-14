#! /bin/bash

source ./make.bash || exit 1

go test -i $PKGS || exit 1
go test $PKGS  || exit 1

(cd examples && ./build.bash) || exit 1
(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
