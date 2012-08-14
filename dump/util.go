package dump

import (
	"io"
	"os"
)

func ReadAll(in Reader) chan *Frame {
	c := make(chan *Frame)
	go func() {
		for in.Err != io.EOF {
			in.Read()
			c <- &in.Frame
		}
		close(c)
	}()
	return c
}

func ReadAllFiles(files []string, crcEnabled bool) chan *Frame {
	c := make(chan *Frame)
	go func() {
		for _, file := range files {
			f, err := os.Open(file)
			if err != nil {
				panic(err)
			}
			in := NewReader(f, crcEnabled)
			for in.Err != io.EOF {
				in.Read()
				c <- &in.Frame
			}
		}
		close(c)
	}()
	return c
}
