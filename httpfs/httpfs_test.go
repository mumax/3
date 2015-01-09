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

	addr := "http://" + l.Addr().String()
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

func TestAppendRead(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")
	mustPass(t, Mkdir("testdata"))

	data := []byte("hello httpfs\n")
	mustFail(t, Append("testdata/file", data)) // file does not exist yet

	mustPass(t, Touch("testdata/file"))
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, Append("testdata/file", data))
	}

	b, errR := Read("testdata/file")
	if errR != nil {
		t.Error(errR)
	}
	if len(b) != (MANYFILES)*len(data) {
		t.Error(len(b), (MANYFILES+1)*len(data))
	}
}

func TestConcurrentWrite(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")
	mustPass(t, Mkdir("testdata"))
	mustPass(t, Touch("testdata/file"))

	f1 := MustCreate("testdata/file")
	f2 := MustCreate("testdata/file")

	fmt.Fprintln(f1, "a")
	mustPass(t, f1.Flush())

	fmt.Fprintln(f2, "a")
	mustFail(t, f2.Flush())

	for i := 0; i < MANYFILES; i++ {
		fmt.Fprintln(f1, "a")
		mustPass(t, f1.Flush())

		fmt.Fprintln(f2, "a")
		mustFail(t, f2.Flush())
	}

}

func TestAppendSize(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustPass(t, Mkdir("testdata"))

	data := []byte("hello httpfs\n")
	mustFail(t, AppendSize("testdata/file", data, 0)) // file does not exist yet
	mustFail(t, AppendSize("testdata/file", data, 1)) // file does not exist yet

	mustPass(t, Touch("testdata/file"))
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, AppendSize("testdata/file", data, int64(i)*int64(len(data))))
	}

	b, errR := Read("testdata/file")
	if errR != nil {
		t.Error(errR)
	}
	if len(b) != (MANYFILES)*len(data) {
		t.Error(len(b), (MANYFILES+1)*len(data))
	}
}

func TestAppendSizeBad(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustPass(t, Mkdir("testdata"))
	mustPass(t, Touch("testdata/file"))

	data := []byte("hello httpfs\n")
	for i := 0; i < MANYFILES; i++ {
		mustFail(t, AppendSize("testdata/file", data, 3)) // bad size
	}
}

func TestPutRead(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustPass(t, Mkdir("testdata"))

	data := []byte("hello httpfs\n")

	// must pass if file does not yet exist
	for i := 0; i < MANYFILES; i++ {
		mustPass(t, Put("testdata/file", data))
	}

	b, errR := Read("testdata/file")
	if errR != nil {
		t.Error(errR)
	}
	if len(b) != len(data) {
		t.Error(len(b), (MANYFILES+1)*len(data))
	}
}

func TestReaderWriter(t *testing.T) {
	Remove("testdata")
	defer Remove("testdata")

	mustPass(t, Mkdir("testdata"))

	// open file for reading when it's not yet there
	{
		out, errO := Open("testdata/file")
		if errO == nil {
			t.Fail()
		}
		if out != nil {
			t.Fail()
		}
	}

	for i := 0; i < MANYFILES; i++ {

		// create and write to file
		{
			out, errO := Create("testdata/file")
			if errO != nil {
				t.Fail()
			}
			if out == nil {
				t.Fail()
			}

			_, errW := fmt.Fprintln(out, "hello_httpfs")
			if errW != nil {
				t.Fail()
			}

			mustPass(t, out.Close())
		}

		// open file for reading and check content
		{
			f, errO := Open("testdata/file")
			if errO != nil {
				t.Fail()
			}
			if f == nil {
				t.Fail()
			}

			var str string
			_, err := fmt.Fscan(f, &str)

			if err != nil {
				t.Error(err)
			}

			if str != "hello_httpfs" {
				t.Error(str)
			}

			if i == 0 {
				mustPass(t, f.Close()) // it's not needed to close the file

			}
		}
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
