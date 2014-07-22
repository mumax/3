package httpfs

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

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
		return 0, fmt.Errorf(`httpfs read "%v": status %v: "%v"`, f.name, resp.StatusCode, resp.Header.Get(X_ERROR))
	}

	// read response
	defer resp.Body.Close()
	nRead, eRead := resp.Body.Read(p)
	if eRead != nil && eRead != io.EOF {
		return nRead, fmt.Errorf(`httpfs read "%v": "%v"`, f.name, eRead.Error())
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
		return 0, fmt.Errorf(`httpfs write %v: status %v "%v"`, f.name, resp.StatusCode, resp.Header.Get(X_ERROR))
	}
	defer resp.Body.Close()
	nRead := readUInt(resp.Body)
	if nRead < 0 {
		return 0, fmt.Errorf("httpfs write %v: illegal argument", f.name)
	}
	return nRead, err
}

// Close implements io.Closer.
func (f *File) Close() error {
	if f == nil || f.client == nil || f.fd == 0 {
		return fmt.Errorf("invalid argument")
	}
	resp := f.client.do("CLOSE", f.u.String(), nil)
	defer resp.Body.Close()
	// invalidate file to avoid accidental use after close
	f.client = nil
	f.fd = 0
	f.u = url.URL{}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(`httpfs close %v: status %v "%v"`, f.name, resp.StatusCode, resp.Header.Get(X_ERROR))
	} else {
		return nil
	}
}

//TODO: sync
