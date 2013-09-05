package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"math"
	"os"
)

func init() {
	DeclFunc("expect", Expect, "Used for automated tests: checks if a value is close enough to the expected value")
	DeclFunc("fprintln", Fprintln, "Print to file")
	DeclFunc("sign", sign, "Signum function")
	//DeclFunc("LoadFile", LoadFile, "Read .dump file and return contents as array.")
}

// Test if have lies within want +/- maxError,
// and print suited message.
func Expect(msg string, have, want, maxError float64) {
	if math.IsNaN(have) || math.IsNaN(want) || math.Abs(have-want) > maxError {
		log.Fatal(msg, ":", " have: ", have, " want: ", want, "±", maxError)
	} else {
		log.Println(msg, ":", have, "OK")
	}
	// note: we also check "want" for NaN in case "have" and "want" are switched.
}

// Append msg to file. Used to write aggregated output of many simulations in one file.
func Fprintln(filename string, msg ...interface{}) {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0666)
	util.FatalErr(err)
	defer f.Close()
	_, err = fmt.Fprintln(f, msg...)
	util.FatalErr(err)
}

// Read a magnetization state from .dump file.
func LoadFile(fname string) *data.Slice {
	s, _ := data.MustReadFile(fname)
	return s
}

// Download a quantity to host,
// or just return its data when already on host.
func Download(q Getter) *data.Slice {
	buf, recycle := q.Get()
	if recycle {
		defer cuda.RecycleBuffer(buf)
	}
	if buf.CPUAccess() {
		return buf
	} else {
		return buf.HostCopy()
	}
}
