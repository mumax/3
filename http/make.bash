#! /bin/bash
echo "package main" > js.go
echo "const js = \`<script type=\"text/javascript\">" >> js.go
cat script.js >> js.go
echo "</script>\`" >> js.go

go build -v
