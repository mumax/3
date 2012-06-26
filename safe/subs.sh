#! /bin/bash

subs32='s/loat32/loat64/g;'
subs32+='s/FLOAT32/FLOAT64/g;'

#sed $subs32 float32s.go > float64s.go
#sed $subs32 float32s_test.go > float64s_test.go

subsc64='s/Float32/Complex64/g;'
subsc64+='s/float32/complex64/g;'
subsc64+='s/FLOAT32/COMPLEX64/g;'
#sed $subsc64 float32s_test.go > complex64s_test.go
#sed $subsc64 float32s.go > complex64s.go


subsc128='s/omplex64/omplex128/g;'
subsc128+='s/COMPLEX64/COMPLEX128/g;'
sed $subsc128 complex64s.go > complex128s.go
sed $subsc128 complex64s_test.go > complex128s_test.go
