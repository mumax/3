package httpfs

// server-side httpfs code

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

// file action gets its own type to avoid mixing up with other strings
type action string

// httpfs actions, handled at /actionName/ (e.g. /ls/, /mkdir/, ...)
const (
	APPEND action = "append"
	LS     action = "ls"
	MKDIR  action = "mkdir"
	PUT    action = "put"
	READ   action = "read"
	RM     action = "rm"
	TOUCH  action = "touch"
)

// RegisterHandlers sets up the http handlers needed for the httpfs protocol (calling go's http.Handle).
// After RegisterHandlers, http.ListenAndServe may be called.
func RegisterHandlers() {
	m := map[action]handlerFunc{
		APPEND: handleAppend,
		LS:     handleLs,
		MKDIR:  handleMkdir,
		PUT:    handlePut,
		READ:   handleRead,
		RM:     handleRemove,
		TOUCH:  handleTouch,
	}
	for k, v := range m {
		http.HandleFunc("/"+string(k)+"/", newHandler(k, v))
	}
	http.Handle("/fs/", http.StripPrefix("/fs/", http.FileServer(http.Dir("."))))
}

// general handler func for file name, optional URL query, input data and response writer.
type handlerFunc func(fname string, data []byte, w io.Writer, query url.Values) error

func newHandler(prefix action, f handlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		fname := r.URL.Path[len(prefix)+2:] // strip "/prefix/"
		query := r.URL.Query()
		data, err := io.ReadAll(r.Body)

		Log("httpfs req:", prefix, fname, query.Encode(), len(data), "B payload")

		if err != nil {
			Log("httpfs err:", prefix, fname, ":", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		err2 := f(fname, data, w, query)
		if err2 != nil {
			Log("httpfs err:", prefix, fname, ":", err2)
			http.Error(w, err2.Error(), http.StatusInternalServerError)
		}
	}
}

func handleAppend(fname string, data []byte, w io.Writer, q url.Values) error {
	size := int64(-1)
	s := q.Get("size")
	if s != "" {
		var err error
		size, err = strconv.ParseInt(s, 0, 64)
		if err != nil {
			return err
		}
	}
	return localAppend(fname, data, size)
}

func handlePut(fname string, data []byte, w io.Writer, q url.Values) error {
	return localPut(fname, data)
}

func handleLs(fname string, data []byte, w io.Writer, q url.Values) error {
	ls, err := localLs(fname)
	if err != nil {
		return err
	}
	return json.NewEncoder(w).Encode(ls)
}

func handleMkdir(fname string, data []byte, w io.Writer, q url.Values) error {
	return localMkdir(fname)
}

func handleTouch(fname string, data []byte, w io.Writer, q url.Values) error {
	return localTouch(fname)
}

func handleRead(fname string, data []byte, w io.Writer, q url.Values) error {
	b, err := localRead(fname)
	if err != nil {
		return err
	}
	_, err2 := w.Write(b)
	return err2
}

func handleRemove(fname string, data []byte, w io.Writer, q url.Values) error {
	return localRemove(fname)
}
