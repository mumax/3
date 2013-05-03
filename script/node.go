package script

import "fmt"

type node struct {
	typ      nodeType
	tok      *token
	parent   *node
	children []*node
}

func (n *node) addChild(c *node) {
	n.children = append(n.children, c)
	c.parent = n
}

func (n *node) replace(a *node) *node {
	for i, c := range n.parent.children {
		if c == n {
			n.parent.children[i] = a
			a.parent = n.parent
			return a
		}
	}
	panic("replace: bug") // i'm not a child of my parent?
}

func split(n *node, split tokenType, childType nodeType) *node {
	spl := &node{typ: n.typ}

	group := &node{typ: childType}
	for _, c := range n.children {
		if c.tok.Type() == split {
			spl.addChild(group)
			group = &node{typ: childType}
		} else {
			group.addChild(c)
		}
	}
	spl.addChild(group)
	return spl
}

type nodeType int

const (
	ERRnode nodeType = iota
	ROOTnode
	TOKENnode
	STATEMENTnode
)

// rm
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

// rm
var nodeStr = map[nodeType]string{ERRnode: "err", ROOTnode: "root", TOKENnode: "tok", STATEMENTnode: "stmt"}

// rm
func (n nodeType) String() string {
	return nodeStr[n]
}
