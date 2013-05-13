#! /bin/bash

# TODO: vet scripts first
scripts=*.txt
for f in $scripts; do
	../mx3 -f $f|| exit 1;
done;

files=*.go
files=$(echo $files | sed s/doc.go//g)

./build.bash || exit 1

for f in $files; do
	a=$(echo $f | sed s/.go//g)
	./$a -o $a.out -f -s || exit 1;
	rm $a
done;

