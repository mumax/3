#! /bin/bash

scripts=*.txt
for f in $scripts; do
	mx3 -vet $f|| exit 1;
done;
for f in $scripts; do
	mx3 -f $f|| exit 1;
	echo ""
done;

