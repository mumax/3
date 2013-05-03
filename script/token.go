package script

import (
	"fmt"
	"strconv"
	"strings"
	"text/scanner"
	"unicode"
)

type token struct {
	typ itemType
	val string
	scanner.Position
}

func (i token) isEOF() bool {
	return i.typ == EOF || i.typ == EOL
}

func (i token) String() string {
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
