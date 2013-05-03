package script

import (
	"io"
	//"code.google.com/p/mx3/util"
	"fmt"
	"os"
)

func parse(src io.Reader) {

	// parse list of tokens
	root, err := lex(src)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	root.Print()
	fmt.Println()

	root = splitlines(root)
	root.Print()
	fmt.Println()
}

func splitlines(n *node) *node {
	spl := &node{}

	group := &node{}
	for _, c := range n.children {
		if c.tok.Type() == EOL {
			spl.addChild(group)
			group = &node{}
		} else {
			group.addChild(c)
		}
	}

	if len(group.children) != 0 {
		spl.addChild(group)
	}

	return spl
}
