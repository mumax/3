#! /bin/bash

source ./build.bash

(cd examples && ./build.bash)
(cd test && ./run.bash)

go test -i $(PKGS) 
go test $(PKGS) 

#go test -i -compiler=$(gccgo) $(PKGS)
#go test -compiler=$(gccgo) $(PKGS)
