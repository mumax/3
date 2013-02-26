package util

import (
	"flag"
	"log"
	"strconv"
)

// Returns the idx'the command line argument
// after proper parsing and error handling.
// Requires flag.Parse() to be called first.
func IntArg(idx int) int {
	if idx >= flag.NArg() {
		log.Fatalf("need command line flag #%v", idx)
	}
	arg := flag.Arg(idx)
	v, err := strconv.Atoi(arg)
	FatalErr(err, "command line flag:")
	return v
}

// Returns the idx'the command line argument
// after proper parsing and error handling.
// Requires flag.Parse() to be called first.
func FloatArg(idx int) float64 {
	if idx >= flag.NArg() {
		log.Fatalf("need command line flag #%v", idx)
	}
	arg := flag.Arg(idx)
	v, err := strconv.ParseFloat(arg, 64)
	FatalErr(err, "command line flag:")
	return v
}
