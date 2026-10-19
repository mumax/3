// Automatic documentation of the data types (Shape, Quantity, ...) that
// appear in the mumax3 API, and of the methods defined on them.
//
// Every named type that can be reached from the script world (as a variable,
// a function argument/return value, or a method argument/return value) is
// collected by reflection. Method docstrings and argument names are taken from
// the Go doc comments in the source of the package that defines the method.
package main

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"html"
	"log"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"unicode"

	"github.com/mumax/3/engine"
)

// A data type as exposed to the script language.
type typeEntry struct {
	T       reflect.Type // type with the largest method set (*T rather than T)
	touched bool
}

var (
	api_types  = make(map[string]*typeEntry) // key: user-visible name (e.g. "Shape")
	goDecls    = make(map[string]*ast.FuncType)
	goDocs     = make(map[string]string) // key: "pkgpath.Type.Method"
	parsedPkgs = make(map[string]bool)
)

func init() { buildTypes() }

func buildTypes() {
	for _, v := range engine.World.Identifiers {
		collectType(v.Type())
	}
}

// collectType registers t (and the types reachable through its methods).
func collectType(t reflect.Type) {
	switch t.Kind() {
	case reflect.Slice, reflect.Array:
		if t.Name() == "" {
			collectType(t.Elem())
			return
		}
	case reflect.Func:
		if t.Name() == "" {
			for i := 0; i < t.NumIn(); i++ {
				collectType(t.In(i))
			}
			for i := 0; i < t.NumOut(); i++ {
				collectType(t.Out(i))
			}
			return
		}
	}
	base := t
	if t.Kind() == reflect.Ptr {
		base = t.Elem()
	}
	if !strings.HasPrefix(base.PkgPath(), "github.com/mumax/3/") || base.Name() == "" {
		return // builtin (float64, ...), unnamed or non-mumax type (time.Time, ...)
	}
	name := base.Name()
	if old, ok := api_types[name]; ok {
		if old.T != t && derefType(old.T) != base {
			log.Panicf("typegen: two types with name %v: %v, %v", name, old.T, t)
		}
		if t.NumMethod() <= old.T.NumMethod() {
			return
		}
		old.T = t
	} else {
		api_types[name] = &typeEntry{T: t}
	}
	for _, m := range visibleMethods(t) {
		collectType(m.Type)
	}
}

func derefType(t reflect.Type) reflect.Type {
	if t.Kind() == reflect.Ptr {
		return t.Elem()
	}
	return t
}

// Methods hidden from the data type documentation, in addition to hidden():
// internal (Go/GPU-level) methods that are of no use in an input script.
// Methods with a name ending in "Go", or with func/unsafe.Pointer arguments
// or return values, are hidden as well.
var typeHidden = map[string]bool{
	"EvalTo": true, "AddTo": true, "Child": true, "Fix": true, "Quantity": true,
	"MSlice": true, "Buffer": true, "SetArray": true, "Gpu": true, "Free": true,
	"Disable": true, "Recycle": true, "IsNil": true, "CPUAccess": true,
	"GPUAccess": true, "MemType": true, "Host": true, "Scalars": true,
	"Vectors": true, "Tensors": true, "HostArray": true, "HostList": true,
}

func internalMethod(m reflect.Method) bool {
	if typeHidden[m.Name] || strings.HasSuffix(m.Name, "Go") {
		return true
	}
	t := m.Type
	for i := 0; i < t.NumIn(); i++ {
		if k := t.In(i).Kind(); k == reflect.Func || k == reflect.UnsafePointer {
			return true
		}
	}
	for i := 0; i < t.NumOut(); i++ {
		if k := t.Out(i).Kind(); k == reflect.Func || k == reflect.UnsafePointer {
			return true
		}
	}
	return false
}

// Docstrings for methods that have no Go doc comment in the source.
// Keys are "Type.Method" or (for all types) "Method".
// ponytail: better to add these as doc comments in the Go source, which take precedence.
var methodDocFallback = map[string]string{
	"Average":                "Returns the spatial average over the simulation box.",
	"magnetization.Average":  "Returns the average magnetization over the magnet geometry.",
	"sOneReg.Average":        "Returns the average over the region.",
	"vOneReg.Average":        "Returns the average over the region.",
	"Comp":                   "Returns component c (0=x, 1=y, 2=z) as a scalar quantity.",
	"Region":                 "Returns a quantity equal to this one inside region r and zero outside. Its average is taken over region r only.",
	"IsUniform":              "Returns true if the value is the same in all regions.",
	"Set":                    "Sets the value in all regions, same as assignment with <code>=</code>.",
	"SetRegion":              "Sets the value in one region. The value may be time-dependent, e.g. <code>B_ext.SetRegion(1, vector(0, 0, 0.1*sin(2*pi*1e9*t)))</code>.",
	"GetRegion":              "Returns the current value in the given region.",
	"GetCell":                "Returns the value in cell (ix, iy, iz).",
	"Get":                    "Returns the current value.",
	"HostCopy":               "Returns a copy of the space-dependent data as a Slice.",
	"magnetization.LoadFile": "Loads the magnetization from an .ovf file.",
	"cropped.Average":        "Returns the average over the cropped part.",
	"Slice.Get":              "Returns component comp of cell (ix, iy, iz).",
	"Slice.Set":              "Sets component comp of cell (ix, iy, iz) to value.",
	"Slice.SetScalar":        "Sets the value of cell (ix, iy, iz) of a 1-component Slice.",
	"Slice.SetVector":        "Sets the value of cell (ix, iy, iz) of a 3-component Slice.",
	"Slice.Index":            "Returns the linear index of cell (ix, iy, iz).",
	"Slice.Size":             "Returns the number of cells in x, y and z.",
	"X":                      "Returns the x-component.",
	"Y":                      "Returns the y-component.",
	"Z":                      "Returns the z-component.",
	"Float":                  "Evaluates the expression and returns the number.",
	"Float3":                 "Evaluates the expression and returns the vector.",
}

// visibleMethods returns the exported, non-hidden methods of t.
// The returned method types never include the receiver.
func visibleMethods(t reflect.Type) []reflect.Method {
	var M []reflect.Method
	for i := 0; i < t.NumMethod(); i++ {
		m := t.Method(i)
		if !unicode.IsUpper(rune(m.Name[0])) || hidden(m.Name) || internalMethod(m) {
			continue
		}
		if t.Kind() != reflect.Interface { // strip receiver
			in := make([]reflect.Type, 0, m.Type.NumIn()-1)
			for j := 1; j < m.Type.NumIn(); j++ {
				in = append(in, m.Type.In(j))
			}
			out := make([]reflect.Type, 0, m.Type.NumOut())
			for j := 0; j < m.Type.NumOut(); j++ {
				out = append(out, m.Type.Out(j))
			}
			m.Type = reflect.FuncOf(in, out, m.Type.IsVariadic())
		}
		M = append(M, m)
	}
	return M
}

// ---- template interface

func (e *typeEntry) Name() string { return derefType(e.T).Name() }
func (e *typeEntry) ID() string   { return "type-" + e.Name() }
func (e *typeEntry) Doc() string  { return goDocs[docKey(derefType(e.T), "")] }

// Interfaces (like Quantity) implemented by this type.
func (e *typeEntry) Implements() string {
	var I []string
	for _, n := range sortedTypeNames() {
		it := api_types[n].T
		if it.Kind() == reflect.Interface && derefType(e.T) != it && e.T.Implements(it) {
			I = append(I, typeLink(it))
		}
	}
	return strings.Join(I, ", ")
}

// Types implementing this interface (like Quantity).
func (e *typeEntry) ImplementedBy() string {
	if e.T.Kind() != reflect.Interface {
		return ""
	}
	var I []string
	for _, n := range sortedTypeNames() {
		t := api_types[n].T
		if t.Kind() != reflect.Interface && t.Implements(e.T) {
			I = append(I, typeLink(t))
		}
	}
	return strings.Join(I, ", ")
}

type methodEntry struct {
	Name, Sig, Doc string
}

func (e *typeEntry) Methods() []methodEntry {
	var M []methodEntry
	for _, m := range visibleMethods(e.T) {
		fn, doc := findMethodDoc(e.T, m.Name)
		if doc == "" {
			doc = methodDocFallback[e.Name()+"."+m.Name]
		}
		if doc == "" {
			doc = methodDocFallback[m.Name]
		}
		M = append(M, methodEntry{m.Name, funcSig(m.Type, fn), doc})
	}
	return M
}

// Type returns the named data type and marks it as documented.
func (a *api) Type(name string) *typeEntry {
	t, ok := api_types[name]
	if !ok {
		log.Panicf("api template: unknown data type %q, have %v", name, sortedTypeNames())
	}
	t.touched = true
	return t
}

// All data types not yet documented explicitly in the template.
func (a *api) RemainingTypes() []*typeEntry {
	var T []*typeEntry
	for _, n := range sortedTypeNames() {
		if t := api_types[n]; !t.touched {
			t.touched = true
			T = append(T, t)
		}
	}
	return T
}

// Signature of an API entry with clickable types: "(float64, float64) Shape" for
// functions, " RegionwiseScalar" for variables.
func (e *entry) Sig() string {
	if e.Type.Kind() == reflect.Func && e.Type.Name() == "" {
		return funcSig(e.Type, funcDecl(engine.World.Identifiers[strings.ToLower(e.name)]))
	}
	return " " + typeLink(e.Type)
}

// ---- formatting

// HTML for a function signature (without name). Argument names are taken from
// decl if possible.
func funcSig(t reflect.Type, decl *ast.FuncType) string {
	var names []string
	if decl != nil {
		for _, f := range decl.Params.List {
			for _, n := range f.Names {
				names = append(names, n.Name)
			}
		}
	}
	if len(names) != t.NumIn() {
		names = nil
	}
	args := make([]string, t.NumIn())
	for i := range args {
		at := t.In(i)
		s := ""
		if t.IsVariadic() && i == t.NumIn()-1 {
			s = "..." + typeLink(at.Elem())
		} else {
			s = typeLink(at)
		}
		if names != nil {
			s = names[i] + " " + s
		}
		args[i] = s
	}
	sig := "(" + strings.Join(args, ", ") + ")"
	var outs []string
	for i := 0; i < t.NumOut(); i++ {
		outs = append(outs, typeLink(t.Out(i)))
	}
	switch len(outs) {
	case 0:
	case 1:
		sig += " " + outs[0]
	default:
		sig += " (" + strings.Join(outs, ", ") + ")"
	}
	return sig
}

// HTML for a type, linked to its documentation if available.
func typeLink(t reflect.Type) string {
	switch {
	case t.Kind() == reflect.Ptr && t.Name() == "":
		return typeLink(t.Elem())
	case t.Kind() == reflect.Slice && t.Name() == "":
		return "[]" + typeLink(t.Elem())
	case t.Kind() == reflect.Func && t.Name() == "":
		return "func" + funcSig(t, nil)
	case t.Kind() == reflect.Interface && t.Name() == "" && t.NumMethod() == 0:
		return "any"
	}
	if te, ok := api_types[t.Name()]; ok && derefType(te.T) == t {
		id := te.ID()
		return fmt.Sprintf(`<a class="api-type-link" href="#%v" onclick="openType(event, '%v')">%v</a>`, id, id, t.Name())
	}
	return html.EscapeString(cleanType(t.String()))
}

func sortedTypeNames() []string {
	var N []string
	for n := range api_types {
		N = append(N, n)
	}
	sort.Slice(N, func(i, j int) bool { return strings.ToLower(N[i]) < strings.ToLower(N[j]) })
	return N
}

// ---- Go doc comments

// funcDecl returns the Go declaration of the function behind a script
// identifier (for argument names), or nil if not found.
func funcDecl(x interface{}) *ast.FuncType {
	f, ok := x.(interface{ Pointer() uintptr })
	if !ok {
		return nil
	}
	rf := runtime.FuncForPC(f.Pointer())
	if rf == nil {
		return nil
	}
	name := rf.Name() // e.g. "github.com/mumax/3/engine.Circle"
	i := strings.LastIndex(name, "/") + 1
	j := strings.Index(name[i:], ".")
	if j < 0 {
		return nil
	}
	pkg, fn := name[:i+j], name[i+j+1:]
	parsePkg(pkg)
	return goDecls[pkg+".."+fn]
}

func docKey(t reflect.Type, method string) string {
	return t.PkgPath() + "." + t.Name() + "." + method
}

// findMethodDoc looks up the declaration and doc comment of method name on t,
// following embedded struct fields for promoted methods.
func findMethodDoc(t reflect.Type, name string) (*ast.FuncType, string) {
	t = derefType(t)
	if t.Name() != "" {
		parsePkg(t.PkgPath())
		if decl, ok := goDecls[docKey(t, name)]; ok {
			return decl, goDocs[docKey(t, name)]
		}
	}
	if t.Kind() == reflect.Struct {
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.Anonymous {
				continue
			}
			if _, ok := f.Type.MethodByName(name); !ok {
				if _, ok := reflect.PointerTo(f.Type).MethodByName(name); !ok {
					continue
				}
			}
			if d, doc := findMethodDoc(f.Type, name); d != nil {
				return d, doc
			}
		}
	}
	return nil, ""
}

// parsePkg parses the Go source of a package and records method declarations
// and doc comments (incl. interface methods and type docs).
func parsePkg(path string) {
	if parsedPkgs[path] {
		return
	}
	parsedPkgs[path] = true
	p, err := build.Import(path, ".", build.FindOnly)
	if err != nil {
		log.Println("typegen:", err)
		return
	}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, p.Dir, nil, parser.ParseComments)
	if err != nil {
		log.Println("typegen:", err)
		return
	}
	for _, pkg := range pkgs {
		if strings.HasSuffix(pkg.Name, "_test") {
			continue
		}
		for _, f := range pkg.Files {
			for _, d := range f.Decls {
				switch d := d.(type) {
				case *ast.FuncDecl:
					if d.Recv == nil || len(d.Recv.List) == 0 {
						goDecls[path+".."+d.Name.Name] = d.Type // plain function
						continue
					}
					recv := d.Recv.List[0].Type
					if s, ok := recv.(*ast.StarExpr); ok {
						recv = s.X
					}
					if id, ok := recv.(*ast.Ident); ok {
						k := path + "." + id.Name + "." + d.Name.Name
						goDecls[k] = d.Type
						goDocs[k] = docText(d.Doc)
					}
				case *ast.GenDecl:
					for _, s := range d.Specs {
						ts, ok := s.(*ast.TypeSpec)
						if !ok {
							continue
						}
						doc := ts.Doc
						if doc == nil && len(d.Specs) == 1 {
							doc = d.Doc
						}
						goDocs[path+"."+ts.Name.Name+"."] = docText(doc)
						if it, ok := ts.Type.(*ast.InterfaceType); ok {
							for _, m := range it.Methods.List {
								if ft, ok := m.Type.(*ast.FuncType); ok && len(m.Names) == 1 {
									k := path + "." + ts.Name.Name + "." + m.Names[0].Name
									goDecls[k] = ft
									goDocs[k] = strings.TrimSpace(docText(m.Doc) + " " + docText(m.Comment))
								}
							}
						}
					}
				}
			}
		}
	}
}

func docText(c *ast.CommentGroup) string {
	if c == nil {
		return ""
	}
	return strings.Join(strings.Fields(c.Text()), " ")
}
