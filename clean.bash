#! /bin/bash

rm -frv $GOPATH/pkg/*/code.google.com/p/mx3
rm -frv $GOPATH/bin/mx3
rm -frv $GOPATH/bin/mx3-*
rm -fv cuda/*.ptx cuda/*_wrapper.go
