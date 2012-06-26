#! /bin/bash

subs32='s/loat32/loat64/g;'
subs32+='s/FLOAT32/FLOAT64/g;'
#sed $subs32 float32s.go > float64s.go

sed $subs32 float32s_test.go > float64s_test.go
