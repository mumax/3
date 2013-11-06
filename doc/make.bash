#! /bin/bash
go build -v   || exit 1
./doc -vet || echo no worries
./doc 
rm -rfv *.out/*.dump
