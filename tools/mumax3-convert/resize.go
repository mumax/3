package main

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"log"
	"strconv"
	"strings"
)

func resize(f *data.Slice, arg string) {
	s := parseSize(arg)
	resized := data.Resample(f, s)
	*f = *resized
}

func parseSize(arg string) (size [3]int) {
	words := strings.Split(arg, "x")
	if len(words) != 3 {
		log.Fatal("resize: need N0xN1xN2 argument")
	}
	for i, w := range words {
		v, err := strconv.Atoi(w)
		util.FatalErr(err)
		size[util.SwapIndex(i, len(words))] = v
	}
	return
}
