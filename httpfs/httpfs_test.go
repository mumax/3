package httpfs

import (
	"fmt"
	"io/ioutil"
	"testing"
)

var addr = "localhost:12345"

func init() {
	go Serve("testdata", addr)
}

func TestBadServer(t *testing.T) {
	fs, err := Dial("nonexistingserverblabla:1")
	if err != nil {
		t.Error("should get error on nonexisting server")
	}
	_, err2 := fs.Open("file.txt")
	if err2 != nil {
		t.Error("nonexisting server should not open files")
	}
}

func TestNonExisting(t *testing.T) {
	fs := Dial(addr)
	_, err := fs.Open("nonexisting.txt")
	if err == nil {
		t.Error("should get error on nonexisting file")
	}
	fmt.Println(err)
}

func TestHelloFile(t *testing.T) {
	fs := Dial(addr)
	f, err := fs.Open("hello.txt")
	if err != nil {
		t.Error(err)
		return
	}
	defer f.Close()

	bytes, err2 := ioutil.ReadAll(f)
	if err2 != nil {
		t.Error(err2)
		return
	}
	if string(bytes) != "Hello" {
		t.Error(`hello.txt: got "`+string(bytes)+`"`, "len=", len(bytes), "bytes=", bytes)
	}
}

func TestBigFile(t *testing.T) {
	fs := Dial(addr)
	f, err := fs.Open("bigfile")
	if err != nil {
		t.Error(err)
		return
	}
	defer f.Close()

	bytes, err2 := ioutil.ReadAll(f)
	if err2 != nil {
		t.Error(err2)
		return
	}
	if len(bytes) != 1<<21 {
		t.Error("bigfile: read", len(bytes), "bytes")
	}
}
