package engine

import (
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"io"
)

var (
	bibfile io.WriteCloser
	library map[string]*bibEntry
)

func init() {
	buildLibrary()
}

func initBib() { // inited in engine.InitIO
	if bibfile != nil {
		panic("bib already inited")
	}
	var err error
	bibfile, err = httpfs.Create(OD() + "bib.txt")
	if err != nil {
		panic(err)
	}
	util.FatalErr(err)

	fprintln(bibfile, "The following list are references relevant for your simulation.")
	fprintln(bibfile, "If you use the results of these simulations in any work or publication, we kindly ask you to cite them.\n")

	Refer("vansteenkiste2014") // Make sure that Mumax3 is always referenced
}

func Refer(tag string) {
	bibentry, inLibrary := library[tag]
	if inLibrary && !bibentry.used {
		bibentry.used = true
		bibentry.write(bibfile)
	}
}

type bibEntry struct {
	tag    string
	ref    string
	title  string
	reason string
	used   bool
}

func (b *bibEntry) write(bibfile io.WriteCloser) {
	if bibfile != nil {
		fprintln(bibfile, b.reason+"\n")
		fprintln(bibfile, "\t\""+b.title+"\"")
		fprintln(bibfile, "\t"+b.ref+"\n")
	}
}

func buildLibrary() {

	library = make(map[string]*bibEntry)

	library["vansteenkiste2014"] = &bibEntry{
		tag:    "vansteenkiste2014",
		title:  "The design and verification of mumax3",
		ref:    "AIP Advances 4, 107133 (2014)",
		reason: "Main paper about Mumax3",
		used:   false}

	library["exl2014"] = &bibEntry{
		tag:    "exl2014",
		title:  "LaBonte's method revisited: An effective steepest descent method for micromagnetic energy minimization",
		ref:    "Journal of Applied Physics 115, 17D118 (2014)",
		reason: "Mumax3 uses Exl's minimizer",
		used:   false}
}
