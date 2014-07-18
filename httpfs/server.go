package httpfs

import (
	"io"
	"log"
	"net/http"
	"os"
	"path"
)

const (
	PROTOCOL    = "http://"
	X_OPEN_FLAG = "X-HTTPFS-OpenFlag"
	X_OPEN_PERM = "X-HTTPFS-OpenPerm"
)

func Serve(root, addr string) {
	log.Println("serving", root, "at", addr)
	err := http.ListenAndServe(addr, &fileHandler{path: root}) // don't use DefaultServeMux which redirects some requests behind our back.
	if err != nil {
		panic(err)
	}
}

type fileHandler struct{ path string }

func (f *fileHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL.Path)
	fname := path.Join(f.path, r.URL.Path)
	defer r.Body.Close()

	switch r.Method {
	default:
		http.Error(w, "method not allowed: "+r.Method, http.StatusMethodNotAllowed)
	case "GET":
		f, err := os.Open(fname)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusNotFound) // TODO: others?
			return
		}
		defer f.Close()
		n, err2 := io.Copy(w, f)
		if err2 != nil {
			log.Println("upload", fname, ":", err2.Error())
		}
		log.Println(n, "bytes sent")
	case "PUT":
		f, err := os.OpenFile(fname, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusNotFound) // TODO: others?
			return
		}
		defer f.Close()
		n, err2 := io.Copy(f, r.Body)
		if err2 != nil {
			log.Println("put", fname, ":", err2.Error())
		}
		log.Println(n, "bytes recieved")
	}
}
