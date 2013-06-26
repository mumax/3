#! /bin/bash

source ./make.bash || exit 1

PKGS=$(echo code.google.com/p/mx3/{,tools/mx3-convert,data,draw,prof,engine,mag,script,util,cuda})

go test -i -race $PKGS || exit 1
go test -race $PKGS  || exit 1
go install -race

(cd test && ./run.bash) || exit 1

go install

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
