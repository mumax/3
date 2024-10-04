/*
	Toy interpreter executes scripts or stdin.
*/
package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/mumax/3/v3/script"
	"io"
	"log"
	"os"
)

var debug = flag.Bool("g", false, "print debug output")

var (
	world *script.World
	ps1   string
)

func main() {
	log.SetFlags(0)
	flag.Parse()
	world = script.NewWorld()
	world.Func("exit", exit)
	script.Debug = *debug

	if flag.NArg() > 1 {
		check(fmt.Errorf("need 0 or 1 input files"))
	}

	if flag.NArg() == 1 {
		src, err := os.Open(flag.Arg(0))
		check(err)
		ps1 = ">"
		interpret(src)
	} else {
		ps1 = ""
		interpret(os.Stdin)
	}
}

func interpret(in io.Reader) {
	scanner := bufio.NewScanner(in)
	for scanner.Scan() {
		safecall(scanner.Text())
	}
	check(scanner.Err())
}

func safecall(code string) {
	if code == "" {
		return
	}
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, "panic:", err)
		}
	}()
	tree, err := world.Compile(code)
	if err == nil {
		for _, stmt := range tree.Child() {
			fmt.Println(stmt.Eval())
		}
	} else {
		fmt.Fprintln(os.Stderr, err)
	}

}

func check(e error) {
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}

func exit() {
	os.Exit(0)
}
