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
	scanner.Scanner
	err error
}

func lex(src io.Reader) ([]*node, error) {
	l := newLexer(src)
	var tokens []*node
	for node := l.next(); node.typ != EOF; node = l.next() {
		if l.err != nil {
			return nil, l.err
		}
		if node.typ == ERR {
			return nil, fmt.Errorf("%v: illegal token: %v", node.Position, node.val)
		}
		tokens = append(tokens, node)
	}
	return tokens, nil
}

func newLexer(src io.Reader) *lexer {
	l := new(lexer)
	l.Init(src)
	l.Whitespace = 1<<'\t' | 1<<' '
	l.Scanner.Error = func(s *scanner.Scanner, msg string) {
		l.err = fmt.Errorf("%v: syntax error: %v", l.Scanner.Position, msg)
	}
	return l
}

func (l *lexer) next() *node {
	tok := l.Scan()
	if tok == scanner.EOF {
		return &node{EOF, "", l.Position}
	} else {
		v := l.TokenText()
		return &node{typeof(v), v, l.Position}
	}
}

type node struct {
	typ itemType
	val string
	scanner.Position
}

func (i node) isEOF() bool {
	return i.typ == EOF || i.typ == EOL
}

func (i node) String() string {
	return i.Position.String() + ":\t" + i.val + "\t" + i.typ.String()
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
	IDENT
)

var typString = map[itemType]string{ERR: "ERR", EOF: "EOF", EOL: "EOL", ASSIGN: "=", NUM: "NUM", STRING: "STRING", LPAREN: "(", RPAREN: ")", IDENT: "IDENT", COMMA: ","}

func (i itemType) String() string {
	if str, ok := typString[i]; ok {
		return str
	} else {
		return fmt.Sprint("type", int(i))
	}

}

var typeMap = map[string]itemType{"\n": EOL, ";": EOL, "=": ASSIGN, "(": LPAREN, ")": RPAREN, ",": COMMA}

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
