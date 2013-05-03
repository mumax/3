package script

import "fmt"

type node struct {
	typ      nodeType
	tok      *token
	children []*node
}

func (n *node) addChild(c *node) {
	n.children = append(n.children, c)
}

func (n *node) String() string {
	tok := ""
	if n.tok != nil {
		tok = "[" + n.tok.val + "]"
	}
	children := ""
	if n.children != nil {
		children = fmt.Sprint(n.children)
	}
	return fmt.Sprint("{", n.typ, tok, children, "}")
}

type nodeType int

const (
	ERRnode nodeType = iota
	ROOTnode
	TOKENnode
	STATEMENTnode
)

var nodeStr = map[nodeType]string{ERRnode: "err", ROOTnode: "root", TOKENnode: "tok", STATEMENTnode: "stmt"}

func (n nodeType) String() string {
	return nodeStr[n]
}
