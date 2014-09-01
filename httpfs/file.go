package httpfs

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
)

var illegalArgument = errors.New("illegal argument")

const BUFFER_SIZE = 1 << 20

// A https File implements a subset of os.File's methods.
// Reading and writing a httpfs File is transparently buffered
// to minimize network overhead.
type File struct {
	name   string // local file name passed to Open
	client *Client
	u      url.URL // url to access file on remote machine
	fd     string  // file descriptor on server
	inBuf  *bufio.Reader
	outBuf *bufio.Writer
}

// Read implements io.Reader.
func (f *File) Read(p []byte) (n int, err error) {
	if f.inBuf == nil {
		f.inBuf = bufio.NewReaderSize((*syncFile)(f), BUFFER_SIZE)
	}
	return f.inBuf.Read(p)
}

// wrapper for File that redirects Read,Write to unbuffered implementations
type syncFile File

func (f *syncFile) Read(p []byte) (n int, err error)  { return (*File)(f).syncRead(p) }
func (f *syncFile) Write(p []byte) (n int, err error) { return (*File)(f).syncWrite(p) }

func (f *File) syncRead(p []byte) (n int, err error) {
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
	log.Println("read", nRead, "bytes")
	return nRead, err // passes on EOF
}

// Write implements io.Writer.
func (f *File) Write(p []byte) (n int, err error) {
	if f.outBuf == nil {
		f.outBuf = bufio.NewWriterSize((*syncFile)(f), BUFFER_SIZE)
	}
	return f.outBuf.Write(p)
}

func (f *File) syncWrite(p []byte) (n int, err error) {
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
	if f == nil || f.client == nil || f.fd == "" {
		return pathErr("close", f.name, illegalArgument)
	}
	if f.outBuf != nil {
		f.outBuf.Flush()
	}
	resp := f.client.do("CLOSE", f.u.String(), nil)
	defer resp.Body.Close()
	// invalidate file to avoid accidental use after close
	f.client = nil
	f.fd = ""
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
		var list []*fileInfo
		//io.Copy(os.Stdout, resp.Body); return
		e2 := json.NewDecoder(resp.Body).Decode(&list)
		if e2 != nil {
			return nil, pathErr("readdir", f.name, e2)
		}
		fi = make([]os.FileInfo, len(list))
		for i, l := range list {
			fi[i] = l
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
