#! /bin/bash

set -e

scripts=*.mx3

mumax3 -vet $scripts

for g in *.go; do
	go run $g;
done

for f in $scripts; do
	mumax3 -f -http "" $f;
	echo ""
done;


