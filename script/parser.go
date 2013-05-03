package script

import (
	"io"
	//"code.google.com/p/mx3/util"
	"fmt"
	"os"
)

func parse(src io.Reader) {
	nodes, err := lex(src)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	for _, n := range nodes {
		fmt.Println(n)
	}

}
