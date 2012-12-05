package dump

import (
	"code.google.com/p/nimble-cube/core"
	"io"
	"os"
	"path"
)

// Returns a channel that pipes all frames
// available through the reader.
// Each frame is only valid until the next
// one is taken from the channel;
// they share the same storage.
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

// Returns a channel that pipes all frames
// available in the set of files.
// Each frame is only valid until the next
// one is taken from the channel;
// they share the same storage.
func ReadAllFiles(files []string, crcEnabled bool) chan *Frame {
	c := make(chan *Frame)
	go func() {
		for _, file := range files {
			f, err := os.Open(file)
			if err != nil {
				panic(err)
			}
			in := NewReader(f, crcEnabled)
			in.Read()
			for in.Err != io.EOF {
				c <- &in.Frame
				in.Read()
			}
			f.Close()
		}
		close(c)
	}()
	return c
}

// Quick-and-dirty dump to a file.
// Useful for debugging.
func Quick(fname string, data [][][][]float32) {
	if path.Ext(fname) == "" {
		fname += ".dump"
	}
	out, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	core.PanicErr(err)
	defer out.Close()
	w := NewWriter(out, CRC_ENABLED)
	w.Header.Components = len(data)
	w.Header.MeshSize = core.SizeOf(data[0])
	w.WriteHeader()
	for i := range data {
		w.WriteData(core.Contiguous(data[i]))
	}
	w.WriteHash()
}

// Quick-and-dirty read from file.
// Returns first frame if there are many.
func ReadFile(fname string) [][][][]float32 {
	if path.Ext(fname) == "" {
		fname += ".dump"
	}
	out, err := os.OpenFile(fname, os.O_RDONLY, 0666)
	core.PanicErr(err)
	defer out.Close()
	r := NewReader(out, CRC_ENABLED)
	core.Fatal(r.Read())
	return r.Tensors()
}
