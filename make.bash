#! /bin/bash

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit
ln -sf $(pwd)/post-commit .git/hooks/post-commit

PKGS=$(echo code.google.com/p/mx3/{,tools/mx3-convert,data,draw,prof,engine,mag,script,util,cuda})

(cd cuda && ./make.bash) || exit 1
(cd web && ./make.bash) || exit 1
go install -v $PKGS || exit 1
go build -v -o mx3 main.go || exit 1
(cd examples && ../mx3 -vet *.txt) || exit 1
(cd examples && ./make.bash) || exit 1

#GCCGO='gccgo -gccgoflags \'-static-libgcc -O4 -Ofast -march=native\''
#go install -v -compiler $GCCGO $PKGS
#go install -v -compiler $GCCGO

