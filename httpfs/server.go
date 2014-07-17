package httpfs

import (
	"io"
	"log"
	"net/http"
	"os"
	"path"
)

func Serve(root, addr string) {
	http.Handle("/", &fileHandler{path: root})
	log.Println("serving", root, "at", addr)
	err := http.ListenAndServe(addr, nil)
	if err != nil {
		panic(err)
	}
}

type fileHandler struct{ path string }

func (f *fileHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL.Path)

	switch r.Method {
	default:
		http.Error(w, "method not allowed: "+r.Method, http.StatusMethodNotAllowed)
	case "GET":
		fname := path.Join(f.path, r.URL.Path)
		f, err := os.Open(fname)
		if err != nil {
			log.Println(err)
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		defer f.Close()
		n, err2 := io.Copy(w, f)
		if err2 != nil {
			log.Println("upload", fname, ":", err2.Error())
		}
		log.Println(n, "bytes uploaded")
	}
}
