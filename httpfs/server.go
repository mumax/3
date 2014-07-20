package httpfs

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
)

type server struct {
	path      string               // served path
	openFiles map[uintptr]*os.File // active file descriptors
}

func Serve(root, addr string) error {
	log.Println("serving", root, "at", addr)
	server := &server{path: root, openFiles: make(map[uintptr]*os.File)}
	err := http.ListenAndServe(addr, server) // don't use DefaultServeMux which redirects some requests behind our back.
	return err
}

func (f *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Println(r.Method, r.URL.Path)
	defer r.Body.Close()

	switch r.Method {
	default:
		http.Error(w, "method not allowed: "+r.Method, http.StatusMethodNotAllowed)
	case "OPEN":
		f.open(w, r)
	}
}

func (f *server) open(w http.ResponseWriter, r *http.Request) {

	// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
	p := path.Clean(r.URL.Path)
	assert(path.IsAbs(p))
	fname := path.Join(f.path, p)

	// parse open flags
	query := r.URL.Query()
	flagStr := query.Get("flag")
	flag, eFlag := strconv.Atoi(flagStr)
	if eFlag != nil {
		http.Error(w, "invalid flag: "+flagStr, http.StatusBadRequest)
		return
	}

	// parse permissions
	permStr := query.Get("perm")
	perm, ePerm := strconv.Atoi(permStr)
	if ePerm != nil {
		http.Error(w, "invalid perm: "+permStr, http.StatusBadRequest)
		return
	}

	// open file, answer with file descriptor
	file, err := os.OpenFile(fname, flag, os.FileMode(perm))
	if err != nil {
		http.Error(w, err.Error(), 400) // TODO: could distinguish: not found, forbidden, ...
		return
	}

	fd := file.Fd()
	f.openFiles[fd] = file
	fmt.Fprint(w, fd)
	log.Println("httpfs: opened", fname, ", fd:", fd)
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}

//func(f*fileHandler) get(){
//f, err := os.Open(fname)
//if err != nil {
//	log.Println(err)
//	http.Error(w, err.Error(), http.StatusNotFound) // TODO: others?
//	return
//}
//defer f.Close()
//n, err2 := io.Copy(w, f)
//if err2 != nil {
//	log.Println("upload", fname, ":", err2.Error())
//}
//log.Println(n, "bytes sent")
//}
