package httpfs

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
)

// An httpfs Client provides access to an httpfs file system.
type Client struct {
	serverAddr string
	client     http.Client
}

// Dial sets up a Client to access files served on addr.
// An error is returned only if addr cannot be resolved by net.ResolveTCPAddr.
// It is not an error if the server is down at the time of Dial.
func Dial(addr string) (*Client, error) {
	if _, err := net.ResolveTCPAddr("tcp", addr); err != nil {
		return nil, fmt.Errorf("httpfs: dial %v: %v", addr, err)
	}
	c := http.Client{}
	fs := &Client{serverAddr: PROTOCOL + addr + "/", client: c}
	return fs, nil
}

// Open file for reading, similar to os.Open.
func (f *Client) Open(fname string) (*File, error) {
	return f.openRead(fname, os.O_RDONLY, 0666)
}

// Open file for writing, unlike os.Create it is write-only.
func (f *Client) Create(fname string) (*File, error) {
	return f.openRead(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
}

// Generalized open call. Most users will use Open or Create instead.
// The specified flag (O_RDONLY etc.) cannot be O_RDWR: httpfs Files are
// either read-only or write-only.
func (f *Client) OpenFile(fname string, flag int, perm os.FileMode) (*File, error) {
	if (flag & os.O_RDWR) != 0 {
		return nil, fmt.Errorf("httpfs: open %v: flag O_RDWR not allowed")
	}
	if (flag & os.O_WRONLY) != 0 {
		return f.openWrite(fname, flag, perm)
	}
	if (flag & os.O_RDONLY) != 0 {
		return f.openRead(fname, flag, perm)
	}
	return nil, fmt.Errorf("httpfs: open %v: invalid flag: %v", flag)
}

func (f *Client) openRead(fname string, flag int, perm os.FileMode) (*File, error) {
	resp, err := f.client.Get(f.fileURL(fname))
	if err != nil {
		panic(err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, fmt.Errorf("httpfs: open %v: server response %v (%s)", fname, resp.StatusCode, readError(resp.Body))
	}
	return &File{name: fname, r: resp.Body}, nil // to be closed by user
}

func (f *Client) openWrite(fname string, flag int, perm os.FileMode) (*File, error) {
	// TODO: sanitize flag

	r, w := io.Pipe()
	req, e1 := http.NewRequest("PUT", f.fileURL(fname), r)
	req.Header.Add(X_OPEN_PERM, fmt.Sprint(perm))
	req.Header.Add(X_OPEN_FLAG, fmt.Sprint(flag))

	if e1 != nil {
		return nil, fmt.Errorf("httpfs: open %v: %v", fname, e1)
	}

	go func() {
		resp, err := f.client.Do(req) //, strings.NewReader("hi"))

		log.Println("c: openwrite err=", err, "resp", resp)
		if resp != nil {
			resp.Body.Close()
		}
	}()
	return &File{name: fname, w: w}, nil
}

func (f *Client) fileURL(name string) string {
	return f.serverAddr + url.QueryEscape(name)
}

// read error message from http.Response.Body
func readError(r io.Reader) []byte {
	B, err := ioutil.ReadAll(r)
	if err != nil {
		log.Println("httpfs client: readerror:", err)
	}
	// strip trailing newline
	if bytes.HasSuffix(B, []byte{'\n'}) {
		B = B[:len(B)-1]
	}
	return B
}
