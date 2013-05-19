package script

// block statement is a list of statements.
type blockStmt []stmt

func (b *blockStmt) append(s stmt) {
	(*b) = append(*b, s)
}

func (b *blockStmt) Exec() {
	for _, s := range *b {
		s.Exec()
	}
}
