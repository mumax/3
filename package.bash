#! /bin/bash

name=mx3.0.8_linux

./test.bash || exit 1
go build -o mx3 || exit 1
(cd examples && go run make.go) || exit 1

rm -f test.log
rm -f cuda/*.ptx
rm -f guide/*.out/*.dump
rm -f *.tar.gz

rm -rf $name*
mkdir ../$name
cp -r ../$name .
mv ../$name .
rm -f $name/TODO
rm -f $name/*commit
rm -f $name/package.bash

tar cv $name | gzip > $name.tar.gz
rm -rf $name

