#! /bin/bash
go run gpu4.go || exit 2
./test4.sh || exit 2
./convolution.sh || exit 2

