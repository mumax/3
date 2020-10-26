package engine

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const DefaultCacheLifetime = 1 * time.Second

var guiplot *TablePlot

// TablePlot is a WriterTo which writes a (cached) plot image of table data.
// The internally cached image will be updated when the column indices have been changed,
// or when the cache lifetime is exceeded.
// If another GO routine is updating the image, the cached image will be written.
type TablePlot struct {
	lock     sync.Mutex
	updating bool

	table      *DataTable
	xcol, ycol int

	cache struct {
		img        []byte        // cached output
		err        error         // cached error
		expirytime time.Time     // expiration time of the cache
		lifetime   time.Duration // maximum lifetime of the cache
	}
}

func NewPlot(table *DataTable) (p *TablePlot) {
	p = &TablePlot{table: table, xcol: 0, ycol: 1}
	p.cache.lifetime = DefaultCacheLifetime
	return
}

func (p *TablePlot) SelectDataColumns(xcolidx, ycolidx int) {
	p.lock.Lock()
	defer p.lock.Unlock()
	if xcolidx != p.xcol || ycolidx != p.ycol {
		p.xcol, p.ycol = xcolidx, ycolidx
		p.cache.expirytime = time.Time{} // this will trigger an update at the next write
	}
}

func (p *TablePlot) WriteTo(w io.Writer) (int64, error) {
	p.update()

	p.lock.Lock()
	defer p.lock.Unlock()

	if p.cache.err != nil {
		return 0, p.cache.err
	}
	nBytes, err := w.Write(p.cache.img)
	return int64(nBytes), err
}

// Updates the cached image if the cache is expired
// Does nothing if the image is already being updated by another GO process
func (p *TablePlot) update() {
	p.lock.Lock()
	xcol, ycol := p.xcol, p.ycol
	needupdate := !p.updating && time.Now().After(p.cache.expirytime)
	p.updating = p.updating || needupdate
	p.lock.Unlock()

	if !needupdate {
		return
	}

	// create plot without the TablePlot being locked!
	img, err := CreatePlot(p.table, xcol, ycol)

	p.lock.Lock()
	p.cache.img, p.cache.err = img, err
	p.updating = false
	if p.xcol == xcol && p.ycol == ycol {
		p.cache.expirytime = time.Now().Add(p.cache.lifetime)
	} else { // column indices have been changed during the update
		p.cache.expirytime = time.Time{}
	}
	p.lock.Unlock()
}

// Returns a png image plot of table data
func CreatePlot(table *DataTable, xcol, ycol int) (img []byte, err error) {
	if table == nil {
		err = errors.New("DataTable pointer is nil")
		return
	}

	data, err := table.Read()
	if err != nil {
		return
	}

	header := table.Header()

	if !(xcol >= 0 && xcol < len(header) && ycol >= 0 && ycol < len(header)) {
		err = errors.New("Invalid column index")
		return
	}

	pl, err := plot.New()
	if err != nil {
		return
	}

	pl.X.Label.Text = header[xcol].Name()
	if unit := header[xcol].Unit(); unit != "" {
		pl.X.Label.Text += " (" + unit + ")"
	}
	pl.Y.Label.Text = header[ycol].Name()
	if unit := header[ycol].Unit(); unit != "" {
		pl.Y.Label.Text += " (" + unit + ")"
	}

	pl.X.Label.Font.SetName("Helvetica")
	pl.Y.Label.Font.SetName("Helvetica")
	pl.X.Label.Padding = 0.2 * vg.Inch
	pl.Y.Label.Padding = 0.2 * vg.Inch

	points := make(plotter.XYs, len(data))
	for i := 0; i < len(data); i++ {
		points[i].X = data[i][xcol]
		points[i].Y = data[i][ycol]
	}

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		return
	}
	scatter.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
	scatter.Shape = draw.CircleGlyph{}
	scatter.Radius = 1
	pl.Add(scatter)

	wr, err := pl.WriterTo(8*vg.Inch, 4*vg.Inch, "png")
	if err != nil {
		return
	}

	buf := bytes.NewBuffer(nil)
	_, err = wr.WriteTo(buf)

	if err != nil {
		return nil, err
	} else {
		return buf.Bytes(), nil
	}
}

func (g *guistate) servePlot(w http.ResponseWriter, r *http.Request) {
	if guiplot == nil {
		guiplot = NewPlot(&Table)
	}

	xcol, errx := strconv.Atoi(strings.TrimSpace(g.StringValue("usingx")))
	ycol, erry := strconv.Atoi(strings.TrimSpace(g.StringValue("usingy")))
	if errx != nil || erry != nil {
		guiplot.SelectDataColumns(-1, -1) // set explicitly invalid column indices
	} else {
		guiplot.SelectDataColumns(xcol, ycol)
	}

	w.Header().Set("Content-Type", "image/png")
	_, err := guiplot.WriteTo(w)

	if err != nil {
		png.Encode(w, image.NewNRGBA(image.Rect(0, 0, 4, 4)))
		g.Set("plotErr", "Plot Error: "+err.Error())
	} else {
		g.Set("plotErr", "")
	}
}
