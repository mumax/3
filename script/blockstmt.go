package script

type blockStmt struct {
	list []stmt
}

func (b *blockStmt) append(s stmt) {
	b.list = append(b.list, s)
}

func (b *blockStmt) Exec() {
	for _, s := range b.list {
		s.Exec()
	}
}
