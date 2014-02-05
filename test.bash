#! /bin/bash

source ./make.bash || exit 1

PKGS=$(echo github.com/mumax/3/{cmd/mumax3,cmd/mumax3-convert,cmd/mumax3-plot,data,draw,prof,engine,mag,script,util,cuda})

go test -i $PKGS || exit 1
go test $PKGS  || exit 1

(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
