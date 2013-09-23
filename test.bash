#! /bin/bash

source ./make.bash || exit 1

PKGS=$(echo github.com/mumax/3/{mumax3,tools/mx3-convert,tools/mx3-plot,data,draw,prof,engine,mag,script,util,cuda})

go test -i $PKGS || exit 1
go test $PKGS  || exit 1

(cd examples && mumax3 -vet *.txt) || exit 1
(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
