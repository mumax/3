package httpfs

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
)

const (
	X_ERROR  = "X-HTTPFS-Error" // HTTP header key for storing errors
	X_HTTPFS = "X-HTTPFS"       // HTTP header identifying response comes from httpfs server, not other
)

type Server struct {
	root string // served path
	sync.Mutex
	openFiles   map[string]*os.File // active file descriptors
	fileServer  http.Handler        // fileserver only handles GET-requests from browser, not httpfs filesystem
	stripPrefix http.Handler
}

// Serve serves the files under directory root at tcp address addr.
func ListenAndServe(root, addr string) error {
	URL, err := url.Parse(addr)
	if err != nil {
		panic(err)
	}
	if URL.Scheme != "http" {
		panic("httpfs: unknown scheme:" + URL.Scheme)
	}

	addr = URL.Host
	prefix := URL.Path

	l, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return Serve(root, l, prefix)
}

func Serve(root string, l net.Listener, prefix string) error {
	Server := NewServer(root, prefix)
	err := http.Serve(l, Server) // don't use DefaultServeMux which redirects some requests behind our back.
	return err
}

// converts http.HandlerFunc to http.Handler, sigh.
type func2Handler struct{ f http.HandlerFunc }

func (h *func2Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) { h.f(w, r) }

// NewServer returns a http handler that serves the fs rooted at given root directory.
func NewServer(root string, prefix string) *Server {
	s := &Server{
		root:       root,
		openFiles:  make(map[string]*os.File),
		fileServer: http.FileServer(http.Dir(root)),
	}

	s.stripPrefix = http.StripPrefix(prefix, &func2Handler{s.serveHTTP})
	return s
}

var methods = map[string]func(*Server, http.ResponseWriter, *http.Request) error{
	"GET":       (*Server).get,
	"OPEN":      (*Server).open,
	"READ":      (*Server).read,
	"WRITE":     (*Server).write,
	"CLOSE":     (*Server).close,
	"MKDIR":     (*Server).mkdir,
	"READDIR":   (*Server).readdir,
	"DELETE":    (*Server).delete,
	"REMOVEALL": (*Server).removeAll,
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.stripPrefix.ServeHTTP(w, r)
}

func (s *Server) serveHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	//log.Println("httpfs server:", r.Method, r.URL)
	//defer log.Println("<<httpfs server done:", r.Method, r.URL)

	//time.Sleep(30*time.Millisecond) // artificial latency for benchmarking

	// crash protection
	defer func() {
		if err := recover(); err != nil {
			log.Println("httpfs panic:", err)
			w.Header().Set(X_ERROR, fmt.Sprint("panic: ", err))
			w.WriteHeader(http.StatusInternalServerError)
		}
	}()

	w.Header().Set(X_HTTPFS, "true")
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
		log.Println("httpfs server error:", err)
	}
}

// servers classical GET requests so the FS is also accessible from browser.
func (s *Server) get(w http.ResponseWriter, r *http.Request) error {
	s.fileServer.ServeHTTP(w, r)
	return nil
}

func (s *Server) LsOF() []string {
	s.Lock()
	defer s.Unlock()
	of := make([]string, 0, len(s.openFiles))
	for k, _ := range s.openFiles {
		of = append(of, k)
	}
	sort.Strings(of)
	return of
}

func (s *Server) NOpenFiles() int {
	s.Lock()
	defer s.Unlock()
	return len(s.openFiles)
}

// by cleaning the (absolute) path, we sandbox it so that ../ can't go above the root export.
func (s *Server) sandboxPath(p string) string {
	p = path.Clean("/" + p)
	assert(path.IsAbs(p))
	return path.Join(s.root, p)
}

// open file, respond with file descriptor
func (s *Server) open(w http.ResponseWriter, r *http.Request) error {
	fname := s.sandboxPath(r.URL.Path)
	flag := intQuery(r, "flag")
	perm := intQuery(r, "perm")

	file, err := os.OpenFile(fname, flag, os.FileMode(perm))
	if err != nil {
		return err
	}
	fd := fileName(r.URL)
	s.storeFD(fd, file) // TODO: close previous if already open
	fmt.Fprint(w, fd)   // respond with file descriptor
	//log.Println("httpfs server OPEN", r.URL.Path, "->", fd)
	return nil
}

func (s *Server) mkdir(w http.ResponseWriter, r *http.Request) error {
	fname := s.sandboxPath(r.URL.Path)
	perm := intQuery(r, "perm")
	return os.Mkdir(fname, os.FileMode(perm))
}

func intQuery(r *http.Request, key string) int {
	str := r.URL.Query().Get(key)
	n, err := strconv.Atoi(str)
	if err != nil {
		log.Println(err) //should not happen
		return 0
	}
	return n
}

func (s *Server) read(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}
	n := intQuery(r, "n")

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
	//log.Println("uploading", nRead, "bytes:", string(buf[:10])+"...")
	_, eUpload := w.Write(buf[:nRead])
	if eUpload != nil {
		log.Println("upload error for read FD", file.Fd(), ":", eUpload)
	}
	return nil
}

func (s *Server) write(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}

	n, err := io.Copy(file, r.Body)
	if err != nil {
		return err
	}
	fmt.Fprint(w, n)
	return nil
}

func (s *Server) close(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}
	s.rmFD(fileName(r.URL))
	return file.Close()
}

func (s *Server) delete(w http.ResponseWriter, r *http.Request) error {
	path := s.sandboxPath(r.URL.Path)
	return os.Remove(path)
}

func (s *Server) removeAll(w http.ResponseWriter, r *http.Request) error {
	path := s.sandboxPath(r.URL.Path)
	return os.RemoveAll(path)
}

func (s *Server) readdir(w http.ResponseWriter, r *http.Request) error {
	file := s.parseFD(r.URL)
	if file == nil {
		return illegalArgument
	}
	fi, err := file.Readdir(intQuery(r, "n"))
	if err != nil {
		return err
	}
	list := make([]fileInfo, len(fi))
	for i, fi := range fi {
		list[i] = fileInfo{
			Nm:   fi.Name(),
			Sz:   fi.Size(),
			Md:   fi.Mode(),
			MdTm: fi.ModTime(),
		}
	}
	return json.NewEncoder(w).Encode(list)
}

func (s *Server) CloseAll(prefix string) {
	s.Lock()
	defer s.Unlock()

	var files []string
	for name, file := range s.openFiles {
		if strings.HasPrefix(name, prefix) {
			log.Println("httpfs server: close dangling file", file)
			files = append(files, name)
			file.Close()
		}
	}
	for _, f := range files {
		delete(s.openFiles, f)
	}
}

func (s *Server) parseFD(URL *url.URL) *os.File {
	return s.getFD(fileName(URL))
}

func fileName(URL *url.URL) string {
	p := URL.Path
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return path.Clean(p)
}

// return suited status code for error
func statusCode(err error) int {
	switch {
	default:
		return 400 // general error
	case err == nil:
		return http.StatusOK
	case err == io.EOF:
		return http.StatusOK // EOF not treated as real error
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
func (s *Server) getFD(fd string) *os.File {
	s.Lock()
	defer s.Unlock()
	return s.openFiles[fd] // TODO: protect against nonexisting FD
}

// thread-safe openFiles[fd] = f
func (s *Server) storeFD(fd string, f *os.File) {
	s.Lock()
	defer s.Unlock()
	s.openFiles[fd] = f
}

// thread-safe delete(openFiles,fd)
func (s *Server) rmFD(fd string) {
	s.Lock()
	defer s.Unlock()
	delete(s.openFiles, fd)
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}
