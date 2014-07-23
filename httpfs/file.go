package httpfs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
)

var illegalArgument = errors.New("illegal argument")

// A https File implements a subset of os.File's methods.
type File struct {
	name   string // local file name passed to Open
	client *Client
	u      url.URL // url to access file on remote machine
	fd     uintptr // file descriptor on server
}

// Read implements io.Reader.
func (f *File) Read(p []byte) (n int, err error) {
	// prepare request
	u := f.u // (a copy)
	q := u.Query()
	q.Set("n", fmt.Sprint(len(p))) // number of bytes to read
	u.RawQuery = q.Encode()

	resp := f.client.do("READ", u.String(), nil)
	if resp.StatusCode != http.StatusOK {
		return 0, mkError("read", f.name, resp)
	}

	// read response
	defer resp.Body.Close()
	nRead, eRead := resp.Body.Read(p)
	if eRead != nil && eRead != io.EOF {
		return nRead, pathErr("read", f.name, eRead)
	}
	if resp.Header.Get(X_ERROR) == "EOF" {
		err = io.EOF
	}
	return nRead, err // passes on EOF
}

// Write implements io.Writer.
func (f *File) Write(p []byte) (n int, err error) {
	resp := f.client.do("WRITE", f.u.String(), bytes.NewBuffer(p))
	if resp.StatusCode != http.StatusOK {
		return 0, mkError("write", f.name, resp)
	}
	defer resp.Body.Close()
	nRead := readUInt(resp.Body)
	if nRead < 0 {
		return 0, pathErr("write", f.name, illegalArgument)
	}
	return nRead, err
}

// Close implements io.Closer.
func (f *File) Close() error {
	if f == nil || f.client == nil || f.fd == 0 {
		return pathErr("close", f.name, illegalArgument)
	}
	resp := f.client.do("CLOSE", f.u.String(), nil)
	defer resp.Body.Close()
	// invalidate file to avoid accidental use after close
	f.client = nil
	f.fd = 0
	f.u = url.URL{}
	if resp.StatusCode != http.StatusOK {
		return mkError("close", f.name, resp)
	} else {
		return nil
	}
}

func (f *File) Readdir(n int) (fi []os.FileInfo, err error) {
	resp := f.client.do("READDIR", mkURL(f.u.Host, f.u.Path, "n", n), nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, mkError("readdir", f.name, resp)
	} else {
		e2 := json.NewDecoder(resp.Body).Decode(&fi)
		if e2 != nil {
			return nil, pathErr("readdir", f.name, e2)
		}
		return
	}
}

func pathErr(op, path string, err error) *os.PathError {
	return &os.PathError{
		Op:   "httpfs " + op,
		Path: path,
		Err:  err}

}

//TODO: sync
