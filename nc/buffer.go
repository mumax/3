package nc

//type Buffer struct{
//	In, Out chan([]float32)
//	Array [][]float32
//	nextIn, nextOut int
//}
//
//func MakeBuffer(NSlice, warp int) Buffer{
//	var b Buffer
//	b.Array = make([][]float32, NSlice)
//	list := make([]float32, NSlice*warp)
//	for i := 0; i < NSlice; i++ {
//		b.Array[i] = list[i*warp : (i+1)*warp]
//	}
//	return b
//}
//
//func (b*Buffer)Run(){
//	for{
//		select{
//			case in := <-b.In: 
//				
//		}
//	}
//}
