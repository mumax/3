package gui

// concatenate elements
func cat(s []string) string {
	str := ""
	for _, s := range s {
		str += s + " "
	}
	return str
}
