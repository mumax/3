#! /bin/bash

ln -sf $(pwd)/pre-commit .git/hooks/pre-commit
ln -sf $(pwd)/post-commit .git/hooks/post-commit

(cd cuda && ./make.bash)     || exit 1
(cd engine && ./make.bash)   || exit 1
(cd mumax3 && go install -v) || exit 1
(cd tools/mumax3-convert && go build && go install) || exit 1
(cd test && mumax3 -vet *.txt)                   || exit 1
(cd doc && mumax3 -vet *.in)               || exit 1

