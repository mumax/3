package script

import (
	"fmt"
)

type node struct {
	tok        *token
	typ        nodeType
	prev, next *node
	parent     *node
	children   []*node
}

func (n *node) addChild(c *node) {
	c.next = nil
	if n.NChild() > 0 {
		prev := n.children[n.NChild()-1]
		c.prev = prev
		prev.next = c
	}
	n.children = append(n.children, c)
	c.parent = n
}

func (n *node) NChild() int { return len(n.children) }

func (n *node) token() string {
	if n.tok != nil {
		return n.tok.val
	} else {
		return ""
	}
}

//func (n *node) setChildren(newc []*node) {
//	// orphan old kids for safety
//	for _, c := range n.children {
//		c.parent = nil
//	}
//	// set new ones
//	n.children = newc
//	for _, c := range n.children {
//		c.parent = n
//	}
//}

//func (n *node) replace(a *node) *node {
//	for i, c := range n.parent.children {
//		if c == n {
//			n.parent.children[i] = a
//			a.parent = n.parent
//			return a
//		}
//	}
//	panic("replace: bug") // i'm not a child of my parent?
//}

//
//func (n *node) split(delim tokenType) {
//	n.setChildren(split(n, delim))
//}
//
//func parenise(tokens[]*node){
//
//	dst := &node{}
//
//	for i, tok:=range tokens{
//		if tok. == RPAREN{ panic("unmatched )") }
//
//		if tok.typ == LPAREN{
//			depth := 1
//			for j:=i; j<len(tokens); j++{
//				if tokens[j].typ == LPAREN { depth++ }
//				if tokens[j].typ == RPAREN { depth-- }
//				if depth == 0{ dst.addChild(parensize1(tokens[i:j]))
//					i=j+1
//				}
//			}
//		}else{
//			dst.addChild(tok)
//		}
//	}
//}
//
//func parenisze1(tokens[]*node)*node{
//	pars := &node{typ: PARENnode}
//	pars.setChildren(parenise(tokens))
//	return pars
//}
//
type nodeType int

const (
	ERRnode nodeType = iota
	ROOTnode
	BLOBnode
	TOKENnode
	STATEMENTnode
)

// rm
func (n *node) String() string {
	if n == nil {
		return "nil"
	} else {
		return n.typ.String() + "_" + n.token()
	}
}

var ident = 0

func (n *node) Print() {
	for i := 0; i < ident; i++ {
		fmt.Print("  ")
	}
	fmt.Println(n.String(), "\t\t[p:", n.prev, " n:", n.next, "]")
	ident++
	for _, c := range n.children {
		c.Print()
	}
	ident--
}

// rm
var nodeStr = map[nodeType]string{ERRnode: "err", ROOTnode: "root", TOKENnode: "tok", STATEMENTnode: "stmt", BLOBnode: "blob"}

// rm
func (n nodeType) String() string {
	if str, ok := nodeStr[n]; ok {
		return str
	} else {
		return fmt.Sprint("type", int(n))
	}
}
