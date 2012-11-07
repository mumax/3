package nimble

//import (
//	"fmt"
//	"sync"
//)
//
//var (
//	uniquelock sync.Mutex
//	uniquetags = make(map[string]int)
//)
//
//// UniqueTag replaces tag by a globally unique one
//// by appending a number if necessary.
//func UniqueTag(tag string) string {
//	uniquelock.Lock()
//	defer uniquelock.Unlock()
//
//	t := tag
//	i := 0
//	_, ok := uniquetags[t]
//	for ok {
//		t = fmt.Sprint(tag, i)
//		i++
//		_, ok = uniquetags[t]
//	}
//	uniquetags[t] = 1
//	return t
//}
