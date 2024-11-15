package httpfs

// client-side API

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
)

var wd = "" // working directory, see SetWD

// SetWD sets a "working directory" for the client side,
// prefixed to all relative local paths passed to client functions (Mkdir, Touch, Remove, ...).
// dir may start with "http://", turning local relative client paths into remote paths.
// E.g.:
//
//		http://path -> http://path
//		path/file   -> wd/path/file
//	 /path/file  -> /path/file
func SetWD(dir string) {
	if dir != "" && !strings.HasSuffix(dir, "/") {
		dir = dir + "/"
	}
	wd = dir
}

// Mkdir creates a directory at specified URL.
func Mkdir(URL string) error {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpMkdir(URL)
	} else {
		return localMkdir(URL)
	}
}

// Touch creates an empty file at the specified URL.
func Touch(URL string) error {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpTouch(URL)
	} else {
		return localTouch(URL)
	}
}

// ReadDir reads and returns all file names in the directory at URL.
func ReadDir(URL string) ([]string, error) {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpLs(URL)
	} else {
		return localLs(URL)
	}
}

// Remove removes the file or directory at URL, and all children it may contain.
// Similar to os.RemoveAll.
func Remove(URL string) error {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpRemove(URL)
	} else {
		return localRemove(URL)
	}
}

// Read the entire file and return its contents.
func Read(URL string) ([]byte, error) {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpRead(URL)
	} else {
		return localRead(URL)
	}
}

// Append p to the file given by URL,
// but first assure that the file had the expected size.
// Used to avoid accidental concurrent writes by two processes to the same file.
// Size < 0 disables size check.
func AppendSize(URL string, p []byte, size int64) error {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpAppend(URL, p, size)
	} else {
		return localAppend(URL, p, size)
	}
}

// Append p to the file given by URL.
func Append(URL string, p []byte) error {
	return AppendSize(URL, p, -1)
}

// Create file given by URL and put data from p there.
func Put(URL string, p []byte) error {
	URL = addWorkDir(URL)
	if isRemote(URL) {
		return httpPut(URL, p)
	} else {
		return localPut(URL, p)
	}
}

func isRemote(URL string) bool {
	return strings.HasPrefix(URL, "http://")
}

// prefix wd to URL if URL is a relative file path
// does not start with "/", "http://"
func addWorkDir(URL string) string {
	if isRemote(URL) {
		return URL
	}
	if !path.IsAbs(URL) {
		return wd + URL
	}
	return URL
}

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

// do a http request.
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
	resp, err = io.ReadAll(response.Body)
	err = mkErr(a, URL, err)
	return
}
