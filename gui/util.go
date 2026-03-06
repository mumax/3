package gui

import "strings"

// concatenate elements
func cat(s []string) string {
	var str strings.Builder
	for _, s := range s {
		str.WriteString(s + " ")
	}
	return str.String()
}
