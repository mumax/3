package script

import (
	"io"
	//"code.google.com/p/mx3/util"
	"fmt"
	"os"
)

func parse(src io.Reader) {
	tokens, err := lex(src)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	for _, t := range tokens {
		fmt.Println(t)
	}

	root := &node{typ: ROOTnode}

	statement := &node{typ: STATEMENTnode}
	for _, t := range tokens {
		if t.isEOF() {
			if len(statement.children) != 0 {
				root.addChild(statement)
			}
			statement = &node{typ: STATEMENTnode}
		} else {
			statement.addChild(&node{typ: TOKENnode, tok: t})
		}
	}

	for _, s := range root.children {
		fmt.Println(s)
	}
}
