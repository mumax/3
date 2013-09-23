#! /bin/bash
go build -v   || exit 1
./examples -vet || echo no worries
./examples 
