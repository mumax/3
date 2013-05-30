#! /bin/bash

echo '/*' > doc.go
echo 'This directory contains example input scripts.' >> doc.go

for f in *.txt; do
	echo '' >> doc.go
	echo '' >> doc.go
	g=$(echo $f | sed 's/.txt//g')
	echo "${g^}" >> doc.go
	echo '' >> doc.go
	echo See file examples/$f >> doc.go
	sed 's$/\*$$g' $f | sed 's$\*/$$g' | awk '{print "\t" $0}' >> doc.go
done;

echo '*/' >> doc.go
echo 'package examples' >> doc.go
go fmt
