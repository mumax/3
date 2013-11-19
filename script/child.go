package script

//func Child(e Expr) []Expr {
//	if c, ok := e.(interface {
//		Child() []Expr
//	}); ok {
//		return c.Child()
//	} else {
//		return nil
//	}
//}

type noChildren struct{}

func (n *noChildren) Child() []Expr { return nil }

func Contains(tree, search Expr) bool {
	if tree == search {
		return true
	} else {
		children := tree.Child()
		for _, e := range children {
			if Contains(e, search) {
				return true
			}
		}
	}
	return false
}
