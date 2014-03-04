#! /bin/bash

set -e

source ./make.bash

PKGS=$(echo github.com/mumax/3/{cmd/mumax3-convert,data,draw,prof,engine,mag,script,util,cuda})

go test -i -race $PKGS || exit 1
go test -race $PKGS  || exit 1
go install -race

(cd test && ./run.bash) || exit 1

go install

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
