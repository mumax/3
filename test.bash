#! /bin/bash

source ./build.bash

go test -i $PKGS
go test $PKGS 

(cd examples && ./build.bash)
(cd test && ./run.bash)

#go test -i -compiler=$gccgo $PKGS
#go test -compiler=$gccgo $PKGS
