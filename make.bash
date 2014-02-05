#! /bin/bash

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit
ln -sf $(pwd)/post-commit .git/hooks/post-commit

(cd cuda && ./make.bash)     || exit 1
(cd engine && ./make.bash)   || exit 1
go install -v github.com/mumax/3/cmd/{mumax3,mumax3-convert,mumax3-plot,bootstrap} || exit 1
(cd test && mumax3 -vet *.txt)                   || exit 1
(cd doc && mumax3 -vet *.in)               || exit 1

