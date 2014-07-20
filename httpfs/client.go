package httpfs

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
)

// An httpfs Client provides access to an httpfs file system.
type Client struct {
	serverAddr string // server address
	client     http.Client
}

// Dial sets up a Client to access files served on addr.
// An error is returned only if addr cannot be resolved by net.ResolveTCPAddr.
// It is not an error if the server is down at the time of Dial.
func Dial(addr string) (*Client, error) {
	if _, err := net.ResolveTCPAddr("tcp", addr); err != nil {
		return nil, fmt.Errorf("httpfs: dial %v: %v", addr, err)
	}
	fs := &Client{serverAddr: addr}
	return fs, nil
}

// Open opens a file for reading, similar to os.Open.
func (f *Client) Open(name string) (*File, error) {
	return f.OpenFile(name, os.O_RDONLY, 0)
}

// Create opens a file for reading/writing, similar to os.Create
func (f *Client) Create(name string) (*File, error) {
	return f.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0666)
}

// OpenFile is similar to os.OpenFile. Most users will use Open or Create instead.
func (f *Client) OpenFile(name string, flag int, perm os.FileMode) (*File, error) {

	// prepare URL for OPEN request
	u := url.URL{Scheme: "http", Host: f.serverAddr, Path: name}
	q := u.Query()
	q.Set("flag", fmt.Sprint(flag))
	q.Set("perm", fmt.Sprint(uint32(perm)))
	u.RawQuery = q.Encode()

	// send OPEN request
	req, eReq := http.NewRequest("OPEN", u.String(), nil)
	panicOn(eReq)
	resp, eResp := f.client.Do(req)
	if eResp != nil {
		return nil, fmt.Errorf(`httpfs open "%v": %v`, name, eResp)
	}
	defer resp.Body.Close()

	// read file descriptor
	body := readBody(resp.Body)
	fd, eFd := strconv.Atoi(body)
	if eFd != nil {
		return nil, fmt.Errorf(`httpfs open "%v": server response: "%s"`, name, body)
	}

	// prepare *File
	fdURL := url.URL{Scheme: "http", Host: f.serverAddr, Path: fmt.Sprint(fd)}
	file := &File{client: f, u: fdURL, name: name, fd: uintptr(fd)}
	return file, nil
}

func panicOn(err error) {
	if err != nil {
		panic(err)
	}
}

//func (f *Client) openRead(fname string, flag int, perm os.FileMode) (*File, error) {
//	resp, err := f.client.Get(f.fileURL(fname))
//	if err != nil {
//		panic(err)
//	}
//	if resp.StatusCode != http.StatusOK {
//		defer resp.Body.Close()
//		return nil, fmt.Errorf("httpfs: open %v: server response %v (%s)", fname, resp.StatusCode, readError(resp.Body))
//	}
//	return &File{name: fname, r: resp.Body}, nil // to be closed by user
//}
//
//func (f *Client) openWrite(fname string, flag int, perm os.FileMode) (*File, error) {
//	// TODO: sanitize flag
//
//	r, w := io.Pipe()
//	req, e1 := http.NewRequest("PUT", f.fileURL(fname), r)
//	req.Header.Add(X_OPEN_PERM, fmt.Sprint(perm))
//	req.Header.Add(X_OPEN_FLAG, fmt.Sprint(flag))
//
//	if e1 != nil {
//		return nil, fmt.Errorf("httpfs: open %v: %v", fname, e1)
//	}
//
//	go func() {
//		resp, err := f.client.Do(req) //, strings.NewReader("hi"))
//
//		log.Println("c: openwrite err=", err, "resp", resp)
//		if resp != nil {
//			resp.Body.Close()
//		}
//	}()
//	return &File{name: fname, w: w}, nil
//}

//func (f *Client) fileURL(name string) string {
//	return f.serverAddr + escapeURLPath(name)
//}
//
//func escapeURLPath(path string) string {
//	u := url.URL{Path: path}
//	return u.String()
//}

// read error message from http.Response.Body
func readBody(r io.Reader) string {
	B, err := ioutil.ReadAll(r)
	if err != nil {
		return "<could not read error: " + err.Error() + ">"
	}
	// strip trailing newline
	if bytes.HasSuffix(B, []byte{'\n'}) {
		B = B[:len(B)-1]
	}
	return string(B)
}
