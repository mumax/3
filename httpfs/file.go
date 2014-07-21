package httpfs

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// A https File implements a subset of os.File's methods.
type File struct {
	name   string // local file name passed to Open
	client *Client
	u      url.URL // url to access file on remote machine
	fd     uintptr // file descriptor on server
}

func (f *File) Read(p []byte) (n int, err error) {
	// send READ request
	u := f.u // (a copy)
	q := u.Query()
	q.Set("n", fmt.Sprint(len(p))) // number of bytes to read
	u.RawQuery = q.Encode()
	req, eReq := http.NewRequest("READ", u.String(), nil)
	panicOn(eReq)
	resp, eResp := f.client.client.Do(req)
	if eResp != nil {
		return 0, fmt.Errorf(`httpfs read "%v": %v`, f.name, eResp)
	}

	// read response
	defer resp.Body.Close()
	nRead, eRead := resp.Body.Read(p)
	if resp.StatusCode != http.StatusOK {
		return nRead, fmt.Errorf(`httpfs read "%v": status %v: "%v"`, f.name, resp.StatusCode, resp.Header.Get(X_ERROR))
	}
	return nRead, eRead // passes on EOF
}

func (f *File) Write(p []byte) (n int, err error) {
	// send WRITE request
	req, eReq := http.NewRequest("WRITE", f.u.String(), bytes.NewBuffer(p))
	panicOn(eReq)
	resp, eResp := f.client.client.Do(req)
	if eResp != nil {
		return 0, fmt.Errorf(`httpfs write "%v": %v`, f.name, eResp)
	}

	defer resp.Body.Close()
	body := readBody(resp.Body)
	nRead, eNRead := strconv.Atoi(string(body))
	if eNRead != nil {
		return 0, fmt.Errorf("httpfs write: bad response: %v", eNRead)
	}

	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf(`httpfs write %v: status %v "%v"`, f.name, resp.StatusCode, resp.Header.Get(X_ERROR))
	}
	return nRead, err
}

//
func (f *File) Close() error {
	if f == nil {
		return fmt.Errorf("invalid argument")
	}

	// send CLOSE request
	req, eReq := http.NewRequest("CLOSE", f.u.String(), nil)
	panicOn(eReq)
	resp, eResp := f.client.client.Do(req)
	if eResp != nil {
		return fmt.Errorf(`httpfs close "%v": %v`, f.name, eResp)
	}

	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(`httpfs close %v: status %v "%v"`, f.name, resp.StatusCode, resp.Header.Get(X_ERROR))
	} else {
		return nil
	}
}
