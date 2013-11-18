#! /bin/bash

scripts=*.txt

mumax3 -vet $scripts || exit 1;

for g in *.go; do
	go run $g || exit 1;
done

time (
for f in $scripts; do
	mumax3 -f -http "" $f || exit 1;
	echo ""
done;)


#	mumax3 -sync -f -http "" $f || exit 1;
#	mumax3 -bx 16 -by 16 -bl 1024  -f -http "" $f || exit 1;

