// +build ignore

/*
 This program generates Go wrappers for cuda sources.
 The cuda file should contain exactly one __global__ void.
*/
package main

import (
	"code.google.com/p/mx3/core"
	"flag"
	"fmt"
	"os"
	"text/scanner"
)

func main() {
	flag.Parse()
	for _, fname := range flag.Args() {
		cuda2go(fname)
	}
}

func cuda2go(fname string) {
	// open cuda file
	f, err := os.Open(fname)
	core.Fatal(err)
	defer f.Close()

	// read tokens
	var token []string
	var s scanner.Scanner
	s.Init(f)
	tok := s.Scan()
	for tok != scanner.EOF {
		if !filter(s.TokenText()) {
			token = append(token, s.TokenText())
		}
		tok = s.Scan()
	}

	// find function name and arguments
	funcname := ""
	argstart, argstop := -1, -1
	for i := 3; i < len(token); i++ {
		if token[i] == "__global__" {
			funcname = token[i+2]
			argstart = i + 4
		}
		if argstart > 0 && token[i] == ")" {
			argstop = i + 1
			break
		}
	}
	argl := token[argstart:argstop]

	// isolate individual arguments
	var args [][]string
	start := 0
	for i, a := range argl {
		if a == "," || a == ")" {
			args = append(args, argl[start:i])
			start = i + 1
		}
	}

	// make pointers Go-style
	for i := range args {
		if args[i][1] == "*" {
			args[i] = []string{args[i][0] + "*", args[i][2]}
		}
	}
	wrapgen(fname, funcname, args)
}

func wrapgen(filename, funcname string, args [][]string) {
	fmt.Println("wrapgen", filename, funcname, args)
}

// should token be filtered out of stream?
func filter(token string) bool {
	switch token {
	case "__restrict__":
		return true
	}
	return false
}
