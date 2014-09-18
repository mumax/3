package httpfs

import (
	"log"
	"net/http"
	"testing"
)

const (
	testAddr = "localhost:35371"
	testURL  = "http://" + testAddr + "/"
)

func init() {
	Handle()
	go func() {
		log.Println("serving", testAddr)
		if err := http.ListenAndServe(testAddr, nil); err != nil {
			panic(err)
		}
	}()
}

func TestStatelessMkdir(t *testing.T) {
	if err := Mkdir("testdata/delete/local/this/is/a/dir"); err != nil {
		t.Error(err)
	}
	if err := Mkdir(testURL + "testdata/delete/remote/this/is/a/dir"); err != nil {
		t.Error(err)
	}
	if err := Mkdir(testURL + "../noneofmybusyness"); err != nil {
		t.Error(err)
	}
}

func TestStatelessReadDir(t *testing.T) {

}

func TestStatelessRemove(t *testing.T) {

}

func TestStatelessRead(t *testing.T) {

}

func TestStatelessAppend(t *testing.T) {
	if err := Mkdir(testURL + "testdata/delete/bla/bla/"); err != nil {
		t.Error(err)
	}
	if err := Append(testURL+"testdata/delete/bla/bla/file", []byte("hello world")); err != nil {
		t.Error(err)
	}
	if err := Append(testURL+"testdata/delete/bla/bla/file", []byte("hello world")); err != nil {
		t.Error(err)
	}
}
