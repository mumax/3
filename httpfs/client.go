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
)

type Client struct {
	serverAddr string
	client     http.Client
}

const (
	PROTOCOL = "http://"
	MAXPORT  = 1<<16 - 1
)

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

// Open file for reading.
func (f *Client) Open(fname string) (io.ReadCloser, error) {
	resp, err := f.client.Get(f.fileURL(fname))
	if err != nil {
		panic(err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, fmt.Errorf("httpfs: open %v: server response %v (%s)", fname, resp.StatusCode, readError(resp.Body))
	}
	return resp.Body, nil // to be closed by user
}

func (f *Client) OpenWrite(fname string) (io.WriteCloser, error) {
	log.Println("c: openwrite", fname)
	r, w := io.Pipe()
	req, e1 := http.NewRequest("PUT", f.fileURL(fname), r)

	if e1 != nil {
		return nil, fmt.Errorf("httpfs: openwrite %v: %v", fname, e1)
	}
	go func() {
		log.Println("c: openwrite", "Do")
		r, err := f.client.Do(req) //, strings.NewReader("hi"))
		log.Println("c: openwrite err=", err, "resp", r)
	}()
	return w, nil
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
