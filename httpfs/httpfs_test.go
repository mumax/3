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

func TestBadAddress(t *testing.T) {
	if _, err := Dial("badport:111111"); err == nil {
		t.Error("should get error on invalid port")
	} else {
		fmt.Println(err)
	}

	if _, err := Dial(":abc"); err == nil {
		t.Error("should get error on malformed port")
	} else {
		fmt.Println(err)
	}

	if _, err := Dial(":malformed:"); err == nil {
		t.Error("should get error on malformed address")
	} else {
		fmt.Println(err)
	}

	if _, err := Dial("nonexistingghostserver.blabla:123"); err == nil {
		t.Error("should get lookup error")
	} else {
		fmt.Println(err)
	}
}

//func TestServerGone(t *testing.T) {
//	fs, err := Dial(":1111") // use noexisting server nevertheless
//	if err != nil {
//		t.Error(err)
//	}
//	fs.Open("file.txt")
//}

func TestNonExisting(t *testing.T) {
	fs, _ := Dial(addr)
	_, err := fs.Open("nonexisting.txt")
	if err == nil {
		t.Error("should get error on nonexisting file")
	}
	fmt.Println(err)
}

func TestHelloFile(t *testing.T) {
	fs, _ := Dial(addr)
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
	fs, _ := Dial(addr)
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
