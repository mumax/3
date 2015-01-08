package httpfs

// client-side code

import (
	"bytes"
	"errors"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strings"
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
