package script

func Child(e Expr) []Expr {
	if c, ok := e.(interface {
		Child() []Expr
	}); ok {
		return c.Child()
	} else {
		return nil
	}
}

func Contains(tree, search Expr) bool {
	if tree == search {
		return true
	} else {
		children := Child(tree)
		for _, e := range children {
			if Contains(e, search) {
				return true
			}
		}
	}
	return false
}
