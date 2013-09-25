#! /bin/bash

rm -frv $GOPATH/pkg/*/github.com/mumax/3/*
rm -frv $GOPATH/bin/mumax3
rm -frv $GOPATH/bin/mumax3-*
rm -fv cuda/*.ptx cuda/*_wrapper.go
