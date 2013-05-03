package script

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"text/scanner"
	"unicode"
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
	EOL
	ASSIGN
	NUM
	STRING
	LPAREN
	RPAREN
	COMMA
	DOT
	IDENT
)

var typString = map[itemType]string{ERR: "ERR", EOF: "EOF", EOL: "EOL", ASSIGN: "=", NUM: "NUM", STRING: "STRING", LPAREN: "(", RPAREN: ")", IDENT: "IDENT", COMMA: ",", DOT: "."}

func (i itemType) String() string {
	if str, ok := typString[i]; ok {
		return str
	} else {
		return fmt.Sprint("type", int(i))
	}

}

var typeMap = map[string]itemType{"\n": EOL, ";": EOL, "=": ASSIGN, "(": LPAREN, ")": RPAREN, ",": COMMA, ".": DOT}

func typeof(token string) itemType {
	if t, ok := typeMap[token]; ok {
		return t
	}
	if strings.HasPrefix(token, `"`) {
		return STRING
	}
	if _, err := strconv.ParseFloat(token, 64); err == nil {
		return NUM
	}
	if unicode.IsLetter(rune(token[0])) {
		return IDENT
	}
	return ERR
}
