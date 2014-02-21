package draw

import (
	//"github.com/mumax/3/data"
	"image"
)

func drawArrows(img *image.NRGBA, arr [3][][][]float32, subsample int) {
	//smaller := data.SizeOf(arr[0])
	//in := data.Downsample(arr[:], smaller)
	//	w, h := len(arr[X][0][0]), len(arr[X][0])
	//	d := len(arr[X])
	//	norm := float32(d)
	//	*img = *recycle(img, w, h)
	//	for iy := 0; iy < h; iy++ {
	//		for ix := 0; ix < w; ix++ {
	//			var x, y, z float32 = 0., 0., 0.
	//			for iz := 0; iz < d; iz++ {
	//				x += arr[0][iz][iy][ix]
	//				y += arr[1][iz][iy][ix]
	//				z += arr[2][iz][iy][ix]
	//			}
	//			x /= norm
	//			y /= norm
	//			z /= norm
	//			img.Set(ix, (h-1)-iy, HSLMap(x, y, z))
	//		}
	//	}
}
