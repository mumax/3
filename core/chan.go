package core

type Chan interface{
	Mesh()*Mesh
	NComp()int
	Comp(int)Chan1
}
