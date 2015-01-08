package httpfs

import (
	"fmt"
	"net"
	"net/http"
	"testing"
)

// leaving this many files open is supposed to trigger os error.
const MANYFILES = 1025

// start local httpfs server, and use http://address/ as WD
func init() {
	l, err := net.Listen("tcp", ":")
	if err != nil {
		panic(err)
	}

	addr := "http://" + l.Addr().String() + "/"
	SetWD(addr)

	RegisterHandlers()

	fmt.Println("serving httpfs:", addr)
	go func() {
		if err := http.Serve(l, nil); err != nil {
			panic(err)
		}
	}()
}

func TestMkdirRemove(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustPass(t, Mkdir("testdata"))
	mustFail(t, Mkdir("testdata"))

	// test for closing files (internally)
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, Remove("testdata"))
		mustPass(t, Mkdir("testdata"))
	}
}

func TestMkdir(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustFail(t, Mkdir("testdata/bla/bla"))
	mustPass(t, Mkdir("testdata/"))
	mustPass(t, Mkdir("testdata/bla"))
	mustPass(t, Mkdir("testdata/bla/bla"))
}

func TestTouch(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustFail(t, Touch("testdata/file"))
	mustPass(t, Mkdir("testdata/"))
	mustPass(t, Touch("testdata/file"))

	// test for closing files (internally)
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, Touch("testdata/file"))
	}
}

func TestReaddir(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	s := func(s []string, e error) error { return e }

	mustFail(t, s(ReadDir("testdata")))

	// test for closing files (internally)
	for i := 0; i < MANYFILES; i++ {
		mustFail(t, s(ReadDir("testdata")))
	}

	mustPass(t, Mkdir("testdata/"))
	mustPass(t, Touch("testdata/file1"))
	mustPass(t, Touch("testdata/file2"))
	mustPass(t, Touch("testdata/file3"))

	ls, err := ReadDir("testdata")
	if err != nil {
		t.Error(err)
	}
	if len(ls) != 3 {
		t.Fail()
	}

	// test for closing files (internally)
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, s(ReadDir("testdata")))
	}
}

func TestRemove(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustPass(t, Remove("testdata"))

	// test for closing files (internally)
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, Remove("testdata"))
	}
}

func mustPass(t *testing.T, err error) {
	if err != nil {
		t.Fatal(err)
	}
}

func mustFail(t *testing.T, err error) {
	if err == nil {
		t.Fatal("did not get error")
	}
}
