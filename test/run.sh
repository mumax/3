#! /bin/bash
go build kernel-film.go || exit 2
./kernel-film 1 1
./kernel-film 1 16
./kernel-film 16 1
./kernel-film 16 16
./kernel-film 4 4
go run demag1.go || exit 2
go run demag2.go || exit 2
go run gpu4.go || exit 2
go run gpu4-3d.go || exit 2
#./test4.sh || exit 2
#./convolution.sh || exit 2

