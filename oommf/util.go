package oommf

import (
	"io"
	"strconv"
)

func readLine(in io.Reader) (line string, eof bool) {
	char := readChar(in)
	eof = isEOF(char)

	for !isEndline(char) {
		line += string(byte(char))
		char = readChar(in)
	}
	return line, eof
}

func isEOF(char int) bool {
	return char == -1
}

func isEndline(char int) bool {
	return isEOF(char) || char == int('\n')
}

//// Blocks until all requested bytes are read.
//type fullReader struct{ io.Reader }
//
//func (r fullReader) Read(p []byte) (n int, err error) {
//	return io.ReadFull(r.Reader, p)
//}

// Reads one character from the Reader.
// -1 means EOF.
// Errors are cought and cause panic
func readChar(in io.Reader) int {
	buffer := [1]byte{}
	switch nr, err := in.Read(buffer[0:]); true {
	case nr < 0: // error
		panic(err)
	case nr == 0: // eof
		return -1
	case nr > 0: // ok
		return int(buffer[0])
	}
	panic("unreachable")
}

func atoi(a string) int {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	return i
}

func atof(a string) float64 {
	i, err := strconv.ParseFloat(a, 64)
	if err != nil {
		panic(err)
	}
	return i
}

const (
	X = 0
	Y = 1
	Z = 2
)
