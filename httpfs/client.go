package httpfs

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
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
	fs := &Client{serverAddr: PROTOCOL + addr + "/", client: http.Client{}}
	return fs, nil
}

// Open file for reading.
func (f *Client) Open(name string) (io.ReadCloser, error) {
	resp, err := f.client.Get(f.serverAddr + "/" + name)
	if err != nil {
		panic(err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, fmt.Errorf("httpfs: open %v: server response %v (%s)", name, resp.StatusCode, readError(resp.Body))
	}
	return resp.Body, nil // to be closed by user
}

func (f *Client) Write(name string) io.WriteCloser {
	r, w := io.Pipe()

	go func() {
		resp, err := f.client.Post(name, "data", r)
		if err != nil {
			panic(err)
		}
		log.Println(resp)
	}()

	return w
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
