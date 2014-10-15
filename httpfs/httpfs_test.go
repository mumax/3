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

func TestMkdir(t *testing.T) {
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

func TestReadDir(t *testing.T) {

}

func TestRemove(t *testing.T) {

}

func TestRead(t *testing.T) {

}

func TestPut(t *testing.T) {
	if err := Put(testURL+"testdata/a/file.txt", []byte("hello")); err != nil {
		t.Error(err)
	}
}

func TestAppend(t *testing.T) {
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
