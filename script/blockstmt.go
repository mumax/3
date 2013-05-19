package script

type blockStmt struct {
	list []stmt
}

func (c *blockStmt) append(s stmt) {
	c.list = append(c.list, s)
}
