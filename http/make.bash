#! /bin/bash
echo "package main" > js.go
echo "const js = \`<script>" >> js.go
cat script.js >> js.go
echo "</script>\`" >> js.go

go build -v
