#! /bin/bash

scripts=*.txt

mx3 -vet $scripts || exit 1;

for f in $scripts; do
	mx3 -f -http "" $f || exit 1;
	echo ""
done;

