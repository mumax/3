#! /bin/bash

scripts=*.txt

mumax3 -vet $scripts || exit 1;

time (
for f in $scripts; do
	mumax3 -f -http "" $f || exit 1;
	echo ""
done;)

