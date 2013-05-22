package script

// block statement is a list of statements.
type blockStmt []Stmt

func (b *blockStmt) append(s Stmt) {
	(*b) = append(*b, s)
}

func (b *blockStmt) Exec() {
	for _, s := range *b {
		s.Exec()
	}
}
