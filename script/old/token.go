package script

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

type tokenType int

const (
	ERR tokenType = iota
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

var typString = map[tokenType]string{ERR: "ERR", EOF: "EOF", EOL: "EOL", ASSIGN: "=", NUM: "NUM", STRING: "STRING", LPAREN: "(", RPAREN: ")", IDENT: "IDENT", COMMA: ","}

func (i tokenType) String() string {
	if str, ok := typString[i]; ok {
		return str
	} else {
		return fmt.Sprint("type", int(i))
	}

}

var typeMap = map[string]tokenType{"\n": EOL, ";": EOL, "=": ASSIGN, "(": LPAREN, ")": RPAREN, ",": COMMA}

func typeof(token string) tokenType {
	if token == "" {
		return EOF
	}
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
