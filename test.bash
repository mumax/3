#! /bin/bash

source ./make.bash || exit 1

PKGS=$(echo github.com/mumax/3/{mumax3,tools/mumax3-convert,tools/mumax3-plot,data,draw,prof,engine,mag,script,util,cuda})

go test -i $PKGS || exit 1
go test $PKGS  || exit 1

(cd doc && mumax3 -vet *.txt) || exit 1
(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
