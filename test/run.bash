#! /bin/bash

set -e

scripts=*.mx3

mumax3 -vet $scripts

mumax3 -failfast -cache /tmp -f -http "" $scripts;

for g in *.go; do
	if [ "$g" != "doc.go" ]; then
		echo go run $g;
		go run $g;
	fi
done
