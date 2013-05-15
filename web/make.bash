#! /bin/bash
echo package web > js.go
echo // THIS FILE IS AUTO GENERATED FROM webgui.js >> js.go
echo // EDITING IS FUTILE >> js.go
echo  >> js.go
echo const templText = \` >> js.go
cat gui.html >> js.go
echo \` >> js.go
gofmt -w js.go
go install -v
go vet
