#! /bin/bash
echo package engine > html.go
echo // THIS FILE IS AUTO GENERATED FROM gui.html >> html.go
echo // EDITING IS FUTILE >> html.go
echo  >> html.go
echo const templText = \` >> html.go
cat gui.html >> html.go
echo \` >> html.go
gofmt -w html.go
go install -v
go vet
