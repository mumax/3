package script

import (
	"io"
	//"code.google.com/p/mx3/util"
	"fmt"
	"os"
)

func parse(src io.Reader) {

	// parse list of tokens
	tokens, err := lex(src)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// rm
	for _, t := range tokens {
		fmt.Println(t)
	}

	root := &node{typ: ROOTnode}
	for _, t := range tokens {
		root.addChild(&node{typ: TOKENnode, tok: t})
	}

	root = split(root, EOL, STATEMENTnode)
	for _, s := range root.children {
		fmt.Println(s)
	}

}
