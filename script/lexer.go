package script

import (
	"io"
	"text/scanner"
)

type lexer struct {
	scan scanner.Scanner
}

func newLexer(src io.Reader) *lexer {
	l := new(lexer)
	l.scan.Init(src)
	l.scan.Whitespace = 1<<'\t' | 1<<' '
	return l
}

func (l *lexer) next() item {
	tok := l.scan.Scan()
	if tok == scanner.EOF {
		return item{EOF, ""}
	} else {
		v := l.scan.TokenText()
		return item{typeof(v), v}
	}
}

type item struct {
	typ itemType
	val string
}

func (i item) String() string {
	return i.val + "\t\t" + i.typ.String()
}

type itemType int

const (
	ERR itemType = iota
	EOF
)

var typString = map[itemType]string{ERR: "ERR", EOF: "EOF"}

func (i itemType) String() string {
	return typString[i]
}

func typeof(tok string) itemType {
	return ERR
}
