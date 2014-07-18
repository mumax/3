package httpfs

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"time"
)

var addr = "localhost:12345"

func init() {
	go Serve("testdata", addr)
}

func TestWrite(t *testing.T) {
	fs, e := Dial(addr)
	if e != nil {
		t.Error(e)
	}

	// write current time to file
	w, err := fs.OpenFile("output", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		t.Error(err)
	}
	msg := time.Now().String()
	_, e2 := w.Write([]byte(msg))
	if e2 != nil {
		t.Error(e2)
	}
	w.Close()

	// read back
	{
		f, err := fs.Open("output")
		if err != nil {
			t.Error(err)
		}
		bytes, err2 := ioutil.ReadAll(f)
		if err2 != nil {
			t.Error(err2)
			return
		}
		if string(bytes) != msg {
			t.Error("read", string(bytes), "instead of", msg)
		}
	}
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

var specialFiles = []string{"=", ":", "!", "?", "[", "*", "%"}

func TestSpecialFiles(t *testing.T) {
	fs, _ := Dial(addr)
	for _, f := range specialFiles {
		if _, err := fs.Open(f); err != nil {
			t.Error(err)
		}
	}
}

func TestPath(t *testing.T) {
	fs, _ := Dial(addr)
	if _, err := fs.Open("//hello.txt"); err != nil {
		t.Error(err)
	}
	if _, err := fs.Open("./hello.txt"); err != nil {
		t.Error(err)
	}
}

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
	const SIZE = 1 << 24
	writeZeros("testdata/bigfile", SIZE)

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
	if len(bytes) != SIZE {
		t.Error("bigfile: read", len(bytes), "bytes")
	}
}

// make file of size N
func writeZeros(fname string, N int) {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	zeros := make([]byte, N)
	f.Write(zeros)
}
