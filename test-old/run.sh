#! /bin/bash
go build kernel-film.go || exit 2
./kernel-film 1 1 || exit 2
./kernel-film 1 8 || exit 2
./kernel-film 8 1 || exit 2
./kernel-film 8 8 || exit 2
./kernel-film 4 4 || exit 2
rm kernel-film
go run demag1.go -f || exit 2
go run demag2.go -f || exit 2
go run gpu4.go -f || exit 2
go run gpu4-3d.go -f || exit 2
#./test4.sh || exit 2
#./convolution.sh || exit 2

