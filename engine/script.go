package engine

import (
	"code.google.com/p/mx3/script"
	"io"
)

func RunScript(src io.Reader) {

	p := script.NewParser(src)
	p.AddFloat("t", &Time)
	p.Exec()

}
