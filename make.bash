#! /bin/bash

rm -f mx3
ln -sf $(pwd)/pre-commit .git/hooks/pre-commit
ln -sf $(pwd)/post-commit .git/hooks/post-commit

(cd cuda && ./make.bash) || exit 1
(cd engine && ./make.bash)  || exit 1
go install -v            || exit 1
(cd tools/mx3-convert && go build && go install) || exit 1
(cd tools/mx3-plot && go build && go install) || exit 1
(cd test && mx3 -vet *.txt) || exit 1
(cd examples && mx3 -vet *.txt) || exit 1

