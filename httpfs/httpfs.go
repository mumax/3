/*
Package httpfs provides a (userspace) file system API over http.
httpfs is used by mumax3-server to proved file system access to the compute nodes.

The API is similar to go's os package, but both local file names and URLs may be passed.
When the file "name" starts with "http://", it is treated as a remote file, otherwise
it is local. Hence, the same API is used for local and remote file access.

*/
package httpfs

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
)

var Logging = false // enables logging

const (
	DirPerm  = 0777 // permissions for new directory
	FilePerm = 0666 // permissions for new files
)

func readBody(r io.ReadCloser) string {
	defer r.Close()
	b, err := ioutil.ReadAll(r)
	if err != nil {
		log.Println("readbody:", err)
		return ""
	}
	return string(b)
}

func mkErr(a action, URL string, err error) error {
	if err == nil {
		return nil
	} else {
		return fmt.Errorf("httpfs %v %v: %v", a, URL, err)
	}
}

func localMkdir(fname string) error {
	return os.Mkdir(fname, DirPerm)
}

func localTouch(fname string) error {
	f, err := os.OpenFile(fname, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0666)
	if err == nil {
		f.Close()
	}
	return err
}

func localLs(fname string) ([]string, error) {

	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ls, err2 := f.Readdirnames(-1)
	if err2 != nil {
		return nil, err2
	}
	return ls, nil
}

func localAppend(fname string, data []byte, size int64) error {
	f, err := os.OpenFile(fname, os.O_APPEND|os.O_WRONLY, FilePerm)
	if err != nil {
		return err
	}
	defer f.Close()

	if size >= 0 {
		fi, errFi := f.Stat()
		if errFi != nil {
			return errFi
		}

		if size != fi.Size() {
			return fmt.Errorf(`httpfs: file size mismatch, possible concurrent access. size=%v B, expected=%v B`, fi.Size(), size)
		}
	}

	_, err2 := f.Write(data)
	return err2
}

func localPut(fname string, data []byte) error {
	_ = os.MkdirAll(path.Dir(fname), DirPerm)

	f, err := os.OpenFile(fname, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, FilePerm)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err2 := f.Write(data)
	return err2
}

func localRead(fname string) ([]byte, error) {
	return ioutil.ReadFile(fname)
}

func localRemove(fname string) error {
	return os.RemoveAll(fname)
}

func Log(msg ...interface{}) {
	if Logging {
		log.Println(msg...)
	}
}
