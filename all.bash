./make.bash || exit 1
./test.bash || exit 1
(cd examples && go run make.go) || exit 1
./package.bash || exit 1
