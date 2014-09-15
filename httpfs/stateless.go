package httpfs

/*
import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
)

func Mkdir(URL string) error {
	if isRemote(URL) {
		return httpMkdir(URL)
	} else {
		return localMkdir(URL)
	}
}

func ReadDir(URL string) ([]string, error) {
	if isRemote(URL) {
		return httpLs(URL)
	} else {
		return localLs(URL)
	}
}

func Remove(URL string) error {
	if isRemote(URL) {
		return httpRemove(URL)
	} else {
		return localRemove(URL)
	}
}

func Read(URL string) (io.ReadCloser, error) {
	if isRemote(URL) {
		return httpRead(URL)
	} else {
		return localRead(URL)
	}
}

func Append(URL string) (io.WriteCloser, error) {
	if isRemote(URL) {
		return httpAppend(URL)
	} else {
		return localAppend(URL)
	}
}

func isRemote(URL string) bool {
	return strings.HasPrefix(URL, "http://")
}

// file action gets its own type to avoid mixing up with other strings
type action string

const (
	APPEND action = "append"
	LS     action = "ls"
	MKDIR  action = "mkdir"
	READ   action = "read"
	RM     action = "rm"
)

// server-side

func Handle() {
	m := map[action]handlerFunc{
		APPEND: handleAppend,
		LS:     handleLs,
		MKDIR:  handleMkdir,
		READ:   handleRead,
		RM:     handleRemove,
	}
	for k, v := range m {
		http.HandleFunc(string(k)+"/", newHandler(k, v))
	}
}

type handlerFunc func(fname string, data []byte, w io.Writer) error

func newHandler(prefix action, f handlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		//defer r.Body.Close()
		fname := r.URL.Path[len(prefix)+1:]
		data, err := ioutil.ReadAll(r.Body)

		if err != nil {
			log.Println("httpfs", prefix, fname, ":", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		log.Println("server:", prefix, fname, len(data), "payload")

		err2 := f(fname, data, w)
		if err2 != nil {
			log.Println("httpfs", prefix, fname, ":", err2)
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func handleAppend(fname string, data []byte, w io.Writer) error {
	return localAppend(fname, data)
}

func handleLs(fname string, data []byte, w io.Writer) error {
	ls, err := localLs(fname)
	if err != nil {
		return err
	}
	return json.NewEncoder(w).Encode(ls)
}

func handleMkdir(fname string, data []byte, w io.Writer) error {
	return localMkdir(fname)
}

func handleRead(fname string, data []byte, w io.Writer) error {
	b, err := localRead(fname)
	if err != nil {
		return err
	}
	_, err2 := w.Write(b)
	return err2
}

func handleRemove(fname string, data []byte, w io.Writer) error {
	return localRemove(fname)
}

func do(a action, URL string, body io.Reader) (resp io.ReadCloser, err error) {
	u, err := url.Parse(URL)
	u.Path = string(a) + "/" + u.Path
	response, errR := http.Post(u.String(), "data", body)
	if errR != nil {
		return nil, mkErr(a, URL, errR)
	}
	//defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, mkErr(a, URL, response.Status+":"+readBody(response))
	}
	resp, err = ioutil.ReadAll(response.Body)
	err = mkErr(a, URL, err)
	return
}

func readBody(r io.Reader) string {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		log.Println("readbody:", err)
		return ""
	}
	return string(b)
}

func mkErr(a action, URL string, err error) error {
	if err == nil {
		return nil
	} else {
		return fmt.Errorf("httpfs %v %v: %v", a, URL, err)
	}
}

// client-side, remote server

func httpMkdir(URL string) error {
	_, err := do(MKDIR, URL, nil)
	return err
}

func httpLs(URL string) (ls []string, err error) {
	r, errHTTP := do(LS, URL, nil)
	if errHTTP != nil {
		return nil, errHTTP
	}
	errJSON := json.Unmarshal(r, &ls)
	if errJSON != nil {
		return nil, mkErr(LS, URL, errJSON)
	}
	return ls, nil
}

func httpAppend(URL string) (io.WriteCloser, error) {
	w, err := do(APPEND, URL, data)
	if err != nil {
		return nil, err
	}

}

func httpRead(URL string) (io.ReadCloser, error) {
	return do(READ, URL, nil)
}

func httpRemove(URL string) error {
	_, err := do(RM, URL, nil)
	return err
}

// client-side, local server

func localMkdir(fname string) error { return os.Mkdir(fname, 0777) }

func localLs(fname string) ([]string, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	ls, err2 := f.Readdirnames(-1)
	if err2 != nil {
		return nil, err2
	}
	return ls, nil
}

func localAppend(fname string, data []byte) error {
	f, err := os.OpenFile(fname, os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err2 := f.Write(data)
	return err2
}

func localRead(fname string) (io.ReadCloser, error) {
	return os.Open(fname)
}

func localRemove(fname string) error {
	return os.RemoveAll(fname)
}

*/
