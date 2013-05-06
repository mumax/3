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

	root.do_parens()
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

func (n *node) do_parens() {

	for _, c := range n.children {
		c.do_parens()
	}

	for i := 0; i < n.NChild(); i++ {

		if n.children[i].tok.Type() == RPAREN {

			var j int
			for j = i; j >= 0; j-- {
				if n.children[j].tok.Type() == LPAREN {
					break
				}
			}
			if j < 0 {
				panic("unmatched )")
			}

			par := &node{typ: PARENSnode, parent: n}
			for k := j + 1; k < i; k++ {
				par.addChild(n.children[k])
			}

			var newkids []*node
			newkids = append(newkids, n.children[:j]...)
			newkids = append(newkids, par)
			newkids = append(newkids, n.children[i+1:]...)
			n.children = newkids
			for _, cc := range n.children {
				cc.parent = n
			}
			i = j
		}
	}
}

func (n *node) do_commas() {

}
