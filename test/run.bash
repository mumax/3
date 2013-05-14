#! /bin/bash

#scripts=*.txt
#for f in $scripts; do
#	mx3-vet $f|| exit 1;
#done;
#
#for f in $scripts; do
#	../mx3 -f $f|| exit 1;
#done;

files=*.go
files=$(echo $files | sed s/doc.go//g)

for f in $files; do
	go build $f || exit 1;
done;

for f in $files; do
	a=$(echo $f | sed s/.go//g)
	./$a -http="" -o $a.out -f || exit 1;
	rm $a
done;

