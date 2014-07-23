package httpfs

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"sync"
)

// HTTP header key for storing errors
const X_ERROR = "X-HTTPFS-Error"

type server struct {
	path string // served path
	sync.Mutex
	openFiles map[uintptr]*os.File // active file descriptors
}

// Serve serves the files under directory root at tcp address addr.
func Serve(root, addr string) error {
	log.Println("serving", root, "at", addr)
	server := &server{path: root, openFiles: make(map[uintptr]*os.File)}
	err := http.ListenAndServe(addr, server) // don't use DefaultServeMux which redirects some requests behind our back.
	return err
}

var methods = map[string]func(*server, http.ResponseWriter, *http.Request) error{
	"OPEN":  (*server).open,
	"READ":  (*server).read,
	"WRITE": (*server).write,
	"CLOSE": (*server).close,
	"MKDIR": (*server).mkdir,
	//"READDIR": (*server).readdir,
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	handler := methods[r.Method]
	if handler == nil {
		w.Header().Set(X_ERROR, "method not allowed: "+r.Method)
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	err := handler(s, w, r)
	if err != nil {
		w.Header().Set(X_ERROR, err.Error())
		w.WriteHeader(statusCode(err))
	}
}

// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
func (s *server) sandboxPath(p string) string {
	p = path.Clean(p)
	assert(path.IsAbs(p))
	return path.Join(s.path, p)
}

func (s *server) open(w http.ResponseWriter, r *http.Request) error {
	//log.Println(r.Method, r.URL)

	fname := s.sandboxPath(r.URL.Path)

	// parse open flags
	query := r.URL.Query()
	flagStr := query.Get("flag")
	flag, eFlag := strconv.Atoi(flagStr)
	if eFlag != nil {
		return illegalArgument
	}

	// parse permissions
	permStr := query.Get("perm")
	perm, ePerm := strconv.Atoi(permStr) // TODO: base8 (also client)
	if ePerm != nil {
		return illegalArgument
	}

	// open file, answer with file descriptor
	file, err := os.OpenFile(fname, flag, os.FileMode(perm))
	if err != nil {
		return err
	}
	fd := file.Fd()
	s.storeFD(fd, file)
	//log.Println("httpfs: opened", fname, ", fd:", fd)
	fmt.Fprint(w, fd) // respond with file descriptor
	return nil
}

func (s *server) mkdir(w http.ResponseWriter, r *http.Request) error {
	//log.Println(r.Method, r.URL)

	fname := s.sandboxPath(r.URL.Path)

	// parse permissions
	query := r.URL.Query()
	permStr := query.Get("perm")
	perm, ePerm := strconv.Atoi(permStr) // TODO: base8 (also client)
	if ePerm != nil {
		return illegalArgument
	}

	return os.Mkdir(fname, os.FileMode(perm))
}

func (s *server) read(w http.ResponseWriter, r *http.Request) error {
	// TODO: limit N
	fdStr := r.URL.Path[len("/"):]
	fd, _ := strconv.Atoi(fdStr)
	file := s.getFD(uintptr(fd))
	if file == nil {
		return illegalArgument
	}

	nStr := r.URL.Query().Get("n")
	n, eN := strconv.Atoi(nStr)
	if eN != nil {
		return illegalArgument
	}

	//log.Println("httpfs: read fd", fd, "n=", n)

	// Go http server does not support Trailer,
	// first read into buffer...
	buf := make([]byte, n)
	nRead, err := file.Read(buf)

	// ...so we can put error in header
	if err != nil && err != io.EOF {
		return err
	}
	if err == io.EOF {
		w.Header().Set(X_ERROR, "EOF")
		// But statusOK, EOF not treated as actual error
	}
	// upload error is server error, not client.
	_, eUpload := w.Write(buf[:nRead])
	//log.Println(nUpload, "bytes uploaded, error=", eUpload)
	if eUpload != nil {
		log.Println("ERROR: upload read FD", fd, ":", eUpload)
	}
	return eUpload
}

func (s *server) write(w http.ResponseWriter, r *http.Request) error {
	fdStr := r.URL.Path[len("/"):]
	fd, _ := strconv.Atoi(fdStr)
	file := s.getFD(uintptr(fd))
	if file == nil {
		return illegalArgument
	}

	n, err := io.Copy(file, r.Body)
	if err != nil {
		return illegalArgument
	}
	_, err = fmt.Fprint(w, n)
	return err
}

func (s *server) close(w http.ResponseWriter, r *http.Request) error {
	//log.Println(r.Method, r.URL)

	fdStr := r.URL.Path[len("/"):]
	fd, _ := strconv.Atoi(fdStr)
	file := s.getFD(uintptr(fd))
	s.rmFD(uintptr(fd))
	if file == nil {
		return illegalArgument
	}
	return file.Close()
}

//func(s*server)readdir(w ){
//
//}

// return suited status code for error
func statusCode(err error) int {
	switch {
	default:
		return 400 // general error
	case err == nil:
		return http.StatusOK
	case os.IsNotExist(err):
		return http.StatusNotFound
	case os.IsPermission(err):
		return http.StatusForbidden
	case os.IsExist(err):
		return http.StatusFound
	}
}

// thread-safe openFiles[fd]
func (s *server) getFD(fd uintptr) *os.File {
	s.Lock()
	defer s.Unlock()
	return s.openFiles[fd] // TODO: protect against nonexisting FD
}

// thread-safe openFiles[fd] = f
func (s *server) storeFD(fd uintptr, f *os.File) {
	s.Lock()
	defer s.Unlock()
	s.openFiles[fd] = f
}

// thread-safe delete(openFiles,fd)
func (s *server) rmFD(fd uintptr) {
	s.Lock()
	defer s.Unlock()
	delete(s.openFiles, fd)
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}
