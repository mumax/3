#! /bin/bash
echo package web > html.go
echo // THIS FILE IS AUTO GENERATED FROM webgui.html >> html.go
echo // EDITING IS FUTILE >> html.go
echo  >> html.go
echo const templText = \` >> html.go
cat gui.html >> html.go
echo \` >> html.go
gofmt -w html.go
go install -v
go vet
