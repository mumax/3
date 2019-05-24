#! /bin/bash

CGO_CFLAGS_ALLOW='(-fno-schedule-insns|-malign-double|-ffast-math)'

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit || echo ""
ln -sf $(pwd)/post-commit .git/hooks/post-commit || echo ""

(cd cuda && ./make.bash)  || exit 1
go install -v github.com/mumax/3/cmd/... || exit 1
#go vet github.com/mumax/3/... || echo ""
(cd test && mumax3 -vet *.mx3) || exit 1
#(cd doc && mumax3 -vet *.mx3)  || exit 1

