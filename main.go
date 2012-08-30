package main

import (
	"flag"
"time"
	"fmt"
	"math/rand"
	"os"
)

var (
	flag_dir = flag.String("dir", ".", "directory to watch")
)

func main() {
	dir, err := os.Open(*flag_dir)
	check(err)
	files, err2 := dir.Readdirnames(-1)
	check(err2)
	rand.Seed(time.Now().UnixNano())
	start := rand.Intn(len(files)) // start at random position

	for I := range files {
		i := (I + start) % len(files)
		fmt.Println(files[i])
	}
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
