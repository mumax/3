package mm

type Buffer struct {
	RecvPipe
	SendPipe
	Array           []float32
	nextIn, nextOut int
}

type Pipe struct{
	Data, Recycle chan[]float32
}

type SendPipe struct{
	Send chan<-[]float32
	GetBack <-chan[]float32
}

type RecvPipe struct{
	Recv <-chan[]float32
	SendBack chan<-[]float32
}

//func MakeBuffer(NSlice, warp int) Buffer {
//	var b Buffer
//	b.Array = make([][]float32, NSlice)
//	list := make([]float32, NSlice*warp)
//	for i := 0; i < NSlice; i++ {
//		b.Array[i] = list[i*warp : (i+1)*warp]
//	}
//	return b
//}

func (b*Buffer)Run(){
	for{
		select{
			case in := <-b.In: 
				
		}
	}
}
