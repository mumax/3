package httpfs

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
)

// An httpfs Client provides access to an httpfs file system.
type Client struct {
	serverAddr string // server address
	client     http.Client
}

// Dial sets up a Client to access files served on addr.
func Dial(addr string) (*Client, error) {
	URL, err := url.Parse(addr)
	if err != nil {
		panic(err)
	}
	addr = URL.Host
	if URL.Scheme != "http" {
		return nil, fmt.Errorf("httpfs dial: unsupported scheme: %v (need http)", URL.Scheme)
	}
	prefix := URL.Path

	if _, err := net.ResolveTCPAddr("tcp", addr); err != nil {
		return nil, fmt.Errorf("httpfs: dial %v: %v", addr, err)
	}
	fs := &Client{serverAddr: addr + prefix}

	// test connection by opening root
	f, err := fs.Open("/")
	if err != nil {
		return nil, fmt.Errorf(`httpfs dial "%v": could not stat "/": %v`, addr, err)
	}
	f.Close()

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
	URL := mkURL(f.serverAddr, name, "flag", flag, "perm", uint32(perm))
	resp := f.do("OPEN", URL, nil)
	if resp.StatusCode != http.StatusOK {
		return nil, mkError("open", name, resp)
	}
	defer resp.Body.Close()
	fd := readString(resp.Body)
	if fd < 0 {
		return nil, &os.PathError{Op: "httpfs open", Path: name, Err: illegalArgument}
	}

	// prepare *File
	fdURL := url.URL{Scheme: "http", Host: f.serverAddr, Path: fmt.Sprint(fd)}
	file := &File{client: f, u: fdURL, name: name, fd: fd}
	//runtime.SetFinalizer(file, file.Close())
	return file, nil
}

// Mkdir creates a new directory with the specified name and permission bits. If there is an error, it will be of type *PathError.
func (f *Client) Mkdir(name string, perm os.FileMode) error {
	URL := mkURL(f.serverAddr, name, "perm", uint32(perm))
	resp := f.do("MKDIR", URL, nil)
	return mkError("mkdir", name, resp)
}

// Remove removes the named file or directory. If there is an error, it will be of type *PathError.
func (f *Client) Remove(name string) error {
	URL := mkURL(f.serverAddr, name)
	resp := f.do("DELETE", URL, nil)
	return mkError("remove", name, resp)
}

// RemoveAll is similar to os.RemoveAll.
func (f *Client) RemoveAll(name string) error {
	URL := mkURL(f.serverAddr, name)
	resp := f.do("REMOVEALL", URL, nil)
	return mkError("removeAll", name, resp)
}

// ReadFile reads a file entirely and returns the content, similar to ioutil.ReadFile.
func (f *Client) ReadFile(fname string) ([]byte, error) {
	file, err := f.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ioutil.ReadAll(file)
}

// mkError returns an *os.PathError whose Err field is based on the response status and error message.
// os.IsExist, os.IsNotExist and os.IsPermission can be used on the error.
func mkError(op, name string, resp *http.Response) error {
	var err error
	switch resp.StatusCode {
	default:
		err = fmt.Errorf(`status %v "%v"`, resp.StatusCode, resp.Header.Get(X_ERROR))
	case http.StatusOK:
		return nil
	case http.StatusNotFound:
		err = os.ErrNotExist
	case http.StatusFound:
		err = os.ErrExist
	case http.StatusForbidden:
		err = os.ErrPermission
	}
	return &os.PathError{Op: "httpfs " + op, Path: name, Err: err}
}

// returns "http://host/path?query[0]=query[1]...
func mkURL(host, Path string, query ...interface{}) string {
	Path = path.Clean(Path)
	u := url.URL{Scheme: "http", Host: host, Path: Path}
	q := u.Query()
	for i := 0; i < len(query); i += 2 {
		q.Set(query[i].(string), fmt.Sprint(query[i+1]))
	}
	u.RawQuery = q.Encode()
	return u.String()
}

// do Does a HTTP request. If an error occurs, it returns a fake response
// with status Teapot and the error message in the header.
func (f *Client) do(method string, URL string, body io.Reader) *http.Response {
	req, eReq := http.NewRequest(method, URL, body)
	panicOn(eReq)
	//log.Println("httpfs client", req.Method, req.URL)
	//defer log.Println("<<<client do", req.Method, req.URL)
	resp, eResp := f.client.Do(req)
	if eResp != nil {
		log.Println("do", method, URL, ":", eResp)
		return &http.Response{
			StatusCode: http.StatusTeapot, // indicates that it's not a real HTTP status
			Header:     http.Header{X_ERROR: []string{eResp.Error()}},
			Body:       ioutil.NopCloser(strings.NewReader(eResp.Error()))}
	}
	if resp.Header.Get(X_HTTPFS) == "" {
		log.Println("httpfs client", method, URL, "response did not have X_HTTPFS set")
		return &http.Response{
			StatusCode: http.StatusNotImplemented,
			Header:     http.Header{X_ERROR: []string{"response did not have X_HTTPFS set"}},
			Body:       ioutil.NopCloser(strings.NewReader(""))}
	}
	return resp
}

// TODO: rm
func panicOn(err error) {
	if err != nil {
		panic(err)
	}
}

// read int from body, -1 in case of error
func readUInt(r io.Reader) int {
	v := -1
	fmt.Fscan(r, &v)
	return v
}

//TODO: disconnect, keepalive, close all files on disconnect/reconnect
//TODO: return *os.PathError
