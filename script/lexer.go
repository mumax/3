package script

import (
	"fmt"
	"io"
	"text/scanner"
)

type lexer struct {
	scan scanner.Scanner
	str  string
	typ  tokenType
}

func newLexer(src io.Reader) *lexer {
	l := new(lexer)
	l.scan.Init(src)
	l.scan.Whitespace = 1<<'\t' | 1<<' '
	l.scan.Error = func(s *scanner.Scanner, msg string) {
		l.str = fmt.Sprintf("%v: syntax error: %v", l.scan.Position, msg)
		l.typ = ERR
	}
	return l
}

func (l *lexer) unexpected() fn {
	err := ""
	if l.typ == ERR {
		err = fmt.Sprint(l.scan.Pos(), ":syntax error:", l.str)
	} else {
		err = fmt.Sprint(l.scan.Pos(), ":unexpected:", l.typ, ":", l.str)
	}
	return func() interface{} { return err }
}

func (l *lexer) advance() {
	l.scan.Scan()
	l.str = l.scan.TokenText()
	l.typ = typeof(l.str)
	log("advance:", l.typ, ":", l.str)
}
