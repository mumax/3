/*
Package httpfs provides a (userspace) file system API over http.
httpfs is used by mumax3-server to proved file system access to the compute nodes.

The API is similar to go's os package, but both local file names and URLs may be passed.
When the file "name" starts with "http://", it is treated as a remote file, otherwise
it is local. Hence, the same API is used for local and remote file access.

*/
package httpfs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
)

var (
	Logging = false    // enables logging
	wd      = ""       // working directory, see SetWD
	lock    sync.Mutex // synchronous local FS access to avoid races
)

const (
	DirPerm  = 0777 // permissions for new directory
	FilePerm = 0666 // permissions for new files
)

// SetWD sets a "working directory" for the client side,
// prefixed to all relative local paths passed to client functions (Mkdir, Touch, Remove, ...).
// dir may start with "http://", turning local relative client paths into remote paths.
// E.g.:
// 	http://path -> http://path
// 	path/file   -> wd/path/file
//  /path/file  -> /path/file
func SetWD(dir string) {
	if dir != "" && !strings.HasSuffix(dir, "/") {
		dir = dir + "/"
	}
	wd = dir
}

// Creates the directory at specified URL (or local file),
// creating all needed parent directories as well.
func Mkdir(URL string) error {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpMkdir(URL)
	} else {
		return localMkdir(URL)
	}
}

func Touch(URL string) error {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpTouch(URL)
	} else {
		return localTouch(URL)
	}
}

func ReadDir(URL string) ([]string, error) {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpLs(URL)
	} else {
		return localLs(URL)
	}
}

func Remove(URL string) error {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpRemove(URL)
	} else {
		return localRemove(URL)
	}
}

func Read(URL string) ([]byte, error) {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpRead(URL)
	} else {
		return localRead(URL)
	}
}

// Append p to the file given by URL,
// but first assure that the file had the expected size.
// Used to avoid accidental concurrent writes by two processes to the same file.
func AppendSize(URL string, p []byte, size int64) error {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpAppend(URL, p, size)
	} else {
		return localAppend(URL, p, size)
	}
}

func Append(URL string, p []byte) error {
	return AppendSize(URL, p, -1)
}

func Put(URL string, p []byte) error {
	URL = cleanup(URL)
	if isRemote(URL) {
		return httpPut(URL, p)
	} else {
		return localPut(URL, p)
	}
}

func isRemote(URL string) bool {
	return strings.HasPrefix(URL, "http://")
}

func cleanup(URL string) string {
	if isRemote(URL) {
		return URL
	}
	if !path.IsAbs(URL) {
		return wd + URL
	}
	return URL
}

// TODO: query values
func do(a action, URL string, body []byte, query url.Values) (resp []byte, err error) {
	u, err := url.Parse(URL)
	u.Path = string(a) + path.Clean("/"+u.Path)
	u.RawQuery = query.Encode()
	response, errR := http.Post(u.String(), "data", bytes.NewReader(body))
	if errR != nil {
		return nil, mkErr(a, URL, errR)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, errors.New("do " + u.String() + ":" + response.Status + ":" + readBody(response.Body))
	}
	resp, err = ioutil.ReadAll(response.Body)
	err = mkErr(a, URL, err)
	return
}

func readBody(r io.Reader) string {
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

// client-side, remote server

func httpMkdir(URL string) error {
	_, err := do(MKDIR, URL, nil, nil)
	return err
}

func httpTouch(URL string) error {
	_, err := do(TOUCH, URL, nil, nil)
	return err
}

func httpLs(URL string) (ls []string, err error) {
	r, errHTTP := do(LS, URL, nil, nil)
	if errHTTP != nil {
		return nil, errHTTP
	}
	errJSON := json.Unmarshal(r, &ls)
	if errJSON != nil {
		return nil, mkErr(LS, URL, errJSON)
	}
	return ls, nil
}

func httpAppend(URL string, data []byte, size int64) error {
	var query map[string][]string
	if size >= 0 {
		query = map[string][]string{"size": {fmt.Sprint(size)}}
	}
	_, err := do(APPEND, URL, data, query)
	return err
}

func httpPut(URL string, data []byte) error {
	_, err := do(PUT, URL, data, nil)
	return err
}

func httpRead(URL string) ([]byte, error) {
	return do(READ, URL, nil, nil)
}

func httpRemove(URL string) error {
	_, err := do(RM, URL, nil, nil)
	return err
}

// client-side, local server

func localMkdir(fname string) error {
	lock.Lock()
	defer lock.Unlock()
	return os.Mkdir(fname, DirPerm)
}

func localTouch(fname string) error {
	lock.Lock()
	defer lock.Unlock()
	f, err := os.Create(fname)
	if err != nil {
		f.Close()
	}
	return err
}

func localLs(fname string) ([]string, error) {
	lock.Lock()
	defer lock.Unlock()

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
	lock.Lock()
	defer lock.Unlock()

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
	lock.Lock()
	defer lock.Unlock()
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
	lock.Lock()
	defer lock.Unlock()
	return ioutil.ReadFile(fname)
}

func localRemove(fname string) error {
	lock.Lock()
	defer lock.Unlock()
	return os.RemoveAll(fname)
}

func Log(msg ...interface{}) {
	if Logging {
		log.Println(msg...)
	}
}
