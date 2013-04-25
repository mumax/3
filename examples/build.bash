#! /bin/bash

files=*.go
files=$(echo $files | sed s/doc.go//g)

for f in $files; do
	go build $f || exit 1;
done;

