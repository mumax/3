#! /bin/bash

scripts=*.txt

../3 -vet $scripts || exit 1;

for f in $scripts; do
	../3 -f -http "" $f || exit 1;
	echo ""
done;

