package httpfs

//
//import (
//	"fmt"
//	"io/ioutil"
//	"os"
//	"testing"
//	"time"
//)
//
//var addr = "http://localhost:12345/fs"
//
//func init() {
//	go ListenAndServe("testdata", addr)
//}
//
//func TestDial(t *testing.T) {
//	if _, err := Dial("http://badport:111111"); err == nil {
//		t.Error("should get error on invalid port")
//	} else {
//		fmt.Println(err)
//	}
//
//	if _, err := Dial("http//:abc"); err == nil {
//		t.Error("should get error on malformed port")
//	} else {
//		fmt.Println(err)
//	}
//
//	if _, err := Dial("http//:malformed:"); err == nil {
//		t.Error("should get error on malformed address")
//	} else {
//		fmt.Println(err)
//	}
//
//	if _, err := Dial("http//nonexistingghostserver.blabla:123"); err == nil {
//		t.Error("should get lookup error")
//	} else {
//		fmt.Println(err)
//	}
//
//	if _, err := Dial("nohttp.com:123"); err == nil {
//		t.Error("should get scheme error")
//	} else {
//		fmt.Println(err)
//	}
//}
//
//func TestOpenFile(t *testing.T) {
//	fs, e := Dial(addr)
//	if e != nil {
//		t.Error(e)
//	}
//
//	if f, err := fs.OpenFile("nonexitst", os.O_WRONLY, 0666); f != nil || err == nil {
//		t.Error("opened non-existing file")
//	}
//
//	if f, err := fs.OpenFile("nonexitst", os.O_RDONLY, 0666); f != nil || err == nil {
//		t.Error("opened non-existing file")
//	}
//
//	if f, err := fs.OpenFile("nonexitst", os.O_RDWR, 0666); f != nil || err == nil {
//		t.Error("opened non-existing file")
//	}
//
//	if f, err := fs.OpenFile("hello.txt", os.O_RDONLY, 0666); f == nil || err != nil {
//		t.Error("could not open existing file")
//	}
//
//	os.Remove("testdata/created.txt")
//	if f, err := fs.OpenFile("created.txt", os.O_RDONLY|os.O_CREATE|os.O_TRUNC, 0666); f == nil || err != nil {
//		t.Error("could not create file")
//	}
//}
//
//func TestRead(t *testing.T) {
//	fs, _ := Dial(addr)
//	f, err := fs.Open("hello.txt")
//	if err != nil {
//		t.Error(err)
//		return
//	}
//	defer f.Close()
//
//	bytes, err2 := ioutil.ReadAll(f)
//	if err2 != nil {
//		t.Error(err2)
//		return
//	}
//	if string(bytes) != "Hello" {
//		t.Error(`hello.txt: got "`+string(bytes)+`"`, "len=", len(bytes), "bytes=", bytes)
//	}
//}
//
//func TestWrite(t *testing.T) {
//	fs, e := Dial(addr)
//	if e != nil {
//		t.Error(e)
//	}
//
//	// write current time to file
//	w, err := fs.OpenFile("output", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
//	if err != nil {
//		t.Error(err)
//	}
//	msg := time.Now().String()
//	_, e2 := w.Write([]byte(msg))
//	if e2 != nil {
//		t.Error(e2)
//	}
//	w.Close()
//
//	// read back
//	{
//		f, err := fs.Open("output")
//		if err != nil {
//			t.Error(err)
//		}
//		bytes, err2 := ioutil.ReadAll(f)
//		if err2 != nil {
//			t.Error(err2)
//			return
//		}
//		if string(bytes) != msg {
//			t.Error("read", string(bytes), "instead of", msg)
//		}
//	}
//}
//
//func TestClose(t *testing.T) {
//	fs, e := Dial(addr)
//	if e != nil {
//		t.Error(e)
//	}
//	// Not properly closing so many files gives "too many open files" error
//	for i := 0; i < 1025; i++ {
//		f, err := fs.Open("hello.txt")
//		if err != nil {
//			t.Fatal(err)
//		}
//		if err := f.Close(); err != nil {
//			t.Fatal(err)
//		}
//	}
//	// Test double close
//	f, _ := fs.Open("hello.txt")
//	f.Close()
//	if err := f.Close(); err == nil {
//		t.Error("double close")
//	} else {
//		fmt.Println(err)
//	}
//}
//
//func TestWriteNonexit(t *testing.T) {
//	fs, e := Dial(addr)
//	if e != nil {
//		t.Error(e)
//	}
//	w, err := fs.OpenFile("nonexisting", os.O_WRONLY, 0666)
//	if err == nil {
//		t.Error("opened non-exising file for writing: no error")
//	} else {
//		if !os.IsNotExist(err) {
//			t.Error("should get IsNoExist")
//		}
//		if os.IsExist(err) {
//			t.Error("should not get IsExist")
//		}
//		if os.IsPermission(err) {
//			t.Error("should not get IsPermission")
//		}
//		fmt.Println(err)
//	}
//	if w != nil {
//		t.Error("opened non-exising file for writing: non-nil file")
//	}
//}
//
//func TestNoAccess(t *testing.T) {
//	fs, e := Dial(addr)
//	if e != nil {
//		t.Error(e)
//	}
//	os.Chmod("testdata/ronly", 0)
//	if _, err := fs.OpenFile("ronly", os.O_WRONLY, 1000); err == nil {
//		t.Error("opened read-only file for writing")
//	} else {
//		if !os.IsPermission(err) {
//			t.Error("should get IsPermission")
//		}
//		fmt.Println(err)
//	}
//}
//
//func TestSandbox(t *testing.T) {
//	fs, e := Dial(addr)
//	if e != nil {
//		t.Error(e)
//	}
//	// try to access file outside of served directory
//	if _, err := fs.OpenFile("../file.go", os.O_RDONLY, 0); err == nil {
//		t.Error("escaped from sandbox")
//	} else {
//		fmt.Println(err)
//	}
//}
//
//var specialFiles = []string{"=", ":", "!", "?", "[", "*", "%", " ", `"`}
//
//// Test a few special file names that wreak havoc when improperly escaped.
//func TestSpecialFileNames(t *testing.T) {
//	fs, _ := Dial(addr)
//	for _, f := range specialFiles {
//		if _, err := fs.Open(f); err != nil {
//			t.Error(err)
//		}
//	}
//}
//
//func TestSpecialPath(t *testing.T) {
//	fs, _ := Dial(addr)
//	if _, err := fs.Open("//hello.txt"); err != nil {
//		t.Error(err)
//	}
//	if _, err := fs.Open("./hello.txt"); err != nil {
//		t.Error(err)
//	}
//}
//
//func TestBigFile(t *testing.T) {
//	const SIZE = 1 << 24
//	data := make([]byte, 1024)
//
//	fs, _ := Dial(addr)
//	f, err := fs.Create("bigfile")
//	if err != nil {
//		t.Error(err)
//		return
//	}
//	for n := 0; n < SIZE; n += len(data) {
//		f.Write(data)
//	}
//	f.Close()
//
//	f, _ = fs.Open("bigfile")
//	defer f.Close()
//	bytes, err2 := ioutil.ReadAll(f)
//	if err2 != nil {
//		t.Error(err2)
//		return
//	}
//	if len(bytes) != SIZE {
//		t.Error("bigfile: read", len(bytes), "bytes")
//	}
//}
//
//func TestMkdir(t *testing.T) {
//	fs, _ := Dial(addr)
//	if err := fs.Mkdir("existingdir", 0777); err == nil {
//		t.Fail()
//	} else {
//		if !os.IsExist(err) {
//			t.Fail()
//		}
//		fmt.Println(err)
//	}
//
//	os.Remove("testdata/newdir")
//	if err := fs.Mkdir("newdir", 0777); err != nil {
//		t.Error(err)
//	}
//}
//
//func TestReaddir(t *testing.T) {
//	fs, _ := Dial(addr)
//	dir, err := fs.Open("existingdir")
//	if err != nil {
//		t.Error(err)
//	}
//	list, err2 := dir.Readdir(0)
//	if err2 != nil {
//		t.Error(err2)
//	}
//	if len(list) != 2 {
//		t.Error("readdir .:", list)
//	}
//	// TODO: test all file attributes
//}
//
//func TestRemove(t *testing.T) {
//	fs, _ := Dial(addr)
//	if f, err := fs.Create("removeMe"); err != nil {
//		t.Fatal(err)
//	} else {
//		f.Close()
//	}
//	if err := fs.Remove("removeMe"); err != nil {
//		t.Fatal(err)
//	}
//	if err := fs.Remove("removeMe"); err == nil {
//		t.Fail()
//	} else {
//		//fmt.Println(err)
//		if !os.IsNotExist(err) {
//			t.Fail()
//		}
//	}
//}
//
//func TestRemoveAll(t *testing.T) {
//	fs, _ := Dial(addr)
//	if err := fs.Mkdir("removeDir", 0777); err != nil && !os.IsExist(err) {
//		t.Fatal(err)
//	}
//	if f, err := fs.Create("removeDir/file"); err != nil {
//		t.Fatal(err)
//	} else {
//		f.Close()
//	}
//
//	if err := fs.RemoveAll("removeDir"); err != nil {
//		t.Fatal(err)
//	}
//	if err := fs.RemoveAll("removeDir"); err != nil {
//		t.Error("removeAll(non-existing) should not give error")
//	}
//}
