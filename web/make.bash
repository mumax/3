#! /bin/bash
echo package engine > webjs.go
echo // THIS FILE IS AUTO GENERATED FROM webgui.js >> webjs.go
echo // EDITING IS FUTILE >> webjs.go
echo  >> webjs.go
echo const templText = \` >> webjs.go
cat webgui.js >> webjs.go
echo \` >> webjs.go
gofmt -w webjs.go
go install -v
go vet
