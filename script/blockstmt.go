package script

import "reflect"

// block statement is a list of statements.
type blockStmt []Expr

func (b *blockStmt) append(s Expr) {
	(*b) = append(*b, s)
}

func (b *blockStmt) Eval() interface{} {
	for _, s := range *b {
		s.Eval()
	}
	return nil
}

func (b *blockStmt) Type() reflect.Type {
	return nil
}
