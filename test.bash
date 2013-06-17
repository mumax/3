#! /bin/bash

source ./make.bash || exit 1

PKGS=$(echo code.google.com/p/mx3/{,tools/mx3-convert,data,draw,prof,engine,mag,script,util,cuda})

go test -i $PKGS || exit 1
go test $PKGS  || exit 1

(cd examples && mx3 -vet *.txt) || exit 1
(cd test && ./run.bash) || exit 1

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
