#! /bin/bash
go build -v   || exit 1
./doc -vet || echo no worries
./doc 
./doc -api
rm -rfv *.out/*.dump doc 
