#! /bin/bash

rm -r -v $GOPATH/pkg/*/code.google.com/p/mx3
rm -r -v $GOPATH/bin/mx3
rm -r -v $GOPATH/bin/mx3-*
rm -v cuda/*.ptx cuda/*_wrapper.go
