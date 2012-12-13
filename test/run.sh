#! /bin/bash
go build kernel.go || exit 2
./kernel 1 1
./kernel 1 16
./kernel 16 1
./kernel 16 16
./kernel 4 4
go run demag1.go || exit 2
go run demag2.go || exit 2
go run gpu4.go || exit 2
go run gpu4-3d.go || exit 2
#./test4.sh || exit 2
#./convolution.sh || exit 2

