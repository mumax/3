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
	"strconv"
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

// SetWD sets a "working directory", prefixed to all relative local paths.
// dir may start with "http://", turning local relative paths into remote paths.
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

// file action gets its own type to avoid mixing up with other strings
type action string

const (
	APPEND action = "append"
	LS     action = "ls"
	MKDIR  action = "mkdir"
	PUT    action = "put"
	READ   action = "read"
	RM     action = "rm"
	TOUCH  action = "touch"
)

// server-side

func Handle() {
	m := map[action]handlerFunc{
		APPEND: handleAppend,
		LS:     handleLs,
		MKDIR:  handleMkdir,
		PUT:    handlePut,
		READ:   handleRead,
		RM:     handleRemove,
		TOUCH:  handleTouch,
	}
	for k, v := range m {
		http.HandleFunc("/"+string(k)+"/", newHandler(k, v))
	}
	http.Handle("/fs/", http.StripPrefix("/fs/", http.FileServer(http.Dir("."))))
}

// general handler func for file name, input data and response writer.
type handlerFunc func(fname string, data []byte, w io.Writer, query url.Values) error

func newHandler(prefix action, f handlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		fname := r.URL.Path[len(prefix)+2:] // strip "/prefix/"
		query := r.URL.Query()
		data, err := ioutil.ReadAll(r.Body)

		Log("httpfs req:", prefix, fname, query.Encode(), len(data), "B payload")

		if err != nil {
			Log("httpfs err:", prefix, fname, ":", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		err2 := f(fname, data, w, query)
		if err2 != nil {
			Log("httpfs err:", prefix, fname, ":", err2)
			http.Error(w, err2.Error(), http.StatusInternalServerError)
		}
	}
}

func handleAppend(fname string, data []byte, w io.Writer, q url.Values) error {
	size := int64(-1)
	s := q.Get("size")
	if s != "" {
		var err error
		size, err = strconv.ParseInt(s, 0, 64)
		if err != nil {
			return err
		}
	}
	return localAppend(fname, data, size)
}

func handlePut(fname string, data []byte, w io.Writer, q url.Values) error {
	return localPut(fname, data)
}

func handleLs(fname string, data []byte, w io.Writer, q url.Values) error {
	ls, err := localLs(fname)
	if err != nil {
		return err
	}
	return json.NewEncoder(w).Encode(ls)
}

func handleMkdir(fname string, data []byte, w io.Writer, q url.Values) error {
	return localMkdir(fname)
}

func handleTouch(fname string, data []byte, w io.Writer, q url.Values) error {
	return localTouch(fname)
}

func handleRead(fname string, data []byte, w io.Writer, q url.Values) error {
	b, err := localRead(fname)
	if err != nil {
		return err
	}
	_, err2 := w.Write(b)
	return err2
}

func handleRemove(fname string, data []byte, w io.Writer, q url.Values) error {
	return localRemove(fname)
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
