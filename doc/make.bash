#! /bin/bash
go build -v   || exit 1
./doc -vet 2> /dev/null || echo no worries
./doc 
rm -rfv *.out/*.ovf
