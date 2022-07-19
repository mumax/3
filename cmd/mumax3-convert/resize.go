package main

import (
	"log"
	"strconv"
	"strings"

	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
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
		size[i] = v
	}
	return
}
