#! /bin/bash

name=mx3.0.10

go build -o mx3 || exit 1
#(cd examples && go run make.go) || exit 1

rm -f test.log
rm -f cuda/*.ptx
rm -f examples/*.out/*.dump
rm -rf test/*.out

rm -rf $name*
mkdir ../$name
cp -r * ../$name
mv ../$name .
rm -f $name/TODO
rm -f $name/*commit
rm -f $name/package.bash
rm -f $name/examples/template.html
rm -f $name/examples/make.go
rm -f $name/examples/*.txt
rm -f $name/*.tar.gz

tar cv $name | gzip > $name.tar.gz
rm -rf $name

