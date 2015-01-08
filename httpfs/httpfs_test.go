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
	_ = Remove("testdata")

	mustPass(t, Mkdir("testdata"))
	mustFail(t, Mkdir("testdata"))

	// test for closing files (internally)
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, Remove("testdata"))
		mustPass(t, Mkdir("testdata"))
	}
}

func TestMkdir(t *testing.T) {
	_ = Remove("testdata")

	mustFail(t, Mkdir("testdata/bla/bla"))
	mustPass(t, Mkdir("testdata/"))
	mustPass(t, Mkdir("testdata/bla"))
	mustPass(t, Mkdir("testdata/bla/bla"))
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
