#! /bin/bash

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit
ln -sf $(pwd)/post-commit .git/hooks/post-commit

PKGS=$(echo code.google.com/p/mx3{,/tools/mx3-convert})
echo compiling $PKGS

(cd cuda && make -j8) || exit 1
go install -v $PKGS || exit 1

#GCCGO='gccgo -gccgoflags \'-static-libgcc -O4 -Ofast -march=native\''
#go install -v -compiler $GCCGO $PKGS
#go install -v -compiler $GCCGO

