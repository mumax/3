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

	root = splitLines(root)
	root.Print()
	fmt.Println()
}

func splitLines(n *node) *node {
	split := &node{typ: ROOTnode}

	line := &node{typ: STATEMENTnode}
	for _, c := range n.children {
		if c.tok.Type() == EOL {
			split.addChild(line)
			line = &node{typ: STATEMENTnode}
		} else {
			line.addChild(c)
		}
	}

	// add trailing line (not terminated by ;)
	if len(line.children) != 0 {
		split.addChild(line)
	}

	return split
}
