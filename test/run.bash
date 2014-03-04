#! /bin/bash

set -e

scripts=*.txt

mumax3 -vet $scripts

mumax3 outputformat.txt
mumax3-convert -png testdata/*.omf outputformat.out/*.ovf

for g in *.go; do
	go run $g;
done

time (
for f in $scripts; do
	mumax3 -f -http "" $f;
	echo ""
done;)


