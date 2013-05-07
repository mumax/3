package script

import (
	"fmt"
	"io"
	"text/scanner"
)

// TODO rm
func log(msg ...interface{}) {
	fmt.Println("--", fmt.Sprint(msg...))
}

type lexer struct {
	scan scanner.Scanner
	err  error
	str  string
	typ  tokenType
}

func newLexer(src io.Reader) *lexer {
	l := new(lexer)
	l.scan.Init(src)
	l.scan.Whitespace = 1<<'\t' | 1<<' '
	l.scan.Error = func(s *scanner.Scanner, msg string) {
		l.err = fmt.Errorf("%v: syntax error: %v", l.scan.Position, msg)
	}
	return l
}

func (l *lexer) unexpected() fn {
	err := fmt.Sprint(l.scan.Pos(), ":unexpected:", l.typ, ":", l.str)
	log("err=", err)
	return func() interface{} { return err }
}

func (l *lexer) advance() {
	l.scan.Scan()
	l.str = l.scan.TokenText()
	l.typ = typeof(l.str)
	log("advance:", l.typ, ":", l.str)
}
