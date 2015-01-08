package httpfs

import (
	"fmt"
	"net"
	"net/http"
	"testing"
)

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

func TestMkdir(t *testing.T) {
	_ = Remove("testdata")

	mustPass(t, Mkdir("testdata"))
	mustFail(t, Mkdir("testdata"))
	mustFail(t, Mkdir("testdata"))

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
