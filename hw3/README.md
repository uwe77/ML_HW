# HW3
## Linear SVM
### Data
- C1

---1---

alpha:
[ 1. 0.0667 1. 0. 1. -0. 1. 0. -0. 1. 0. 0. -0. 1. 0. -0. 0.0667 -0.  0.0667 -0. 1. 0. 1.     -0.     -0.     -0.      1.  1. 0. -0.      0.      1.0.      0.     -0.      1.  0.2    -0.      1.     -0.     -0.      0.      0.      0.      1. -0.      1.  0. 1.0.  ]

bias: 10.6272, Classifacation Rate: 96.0

---2---

alpha:
[0.    1.    1.    0.672 0.    0.    0.    0.    1.    0.672 1.    1. 0.    0.    0.    0.    1.    0.    0.    0.    0.  0.    0.    0. 0.    0.    1.    1.    0.    0.    0.    0.    0.    1.    0.344 0. 0.    0.    1.    0.    0.    0.    1.    0.    0.    0.    1.    0. 0.    1.   ]

bias: 12.4949, Classifacation Rate: 96.0

1&2 Classifacation Rate: 96.0
- C10

---1---

alpha:
[ 0.      0.      8.9995  0.      0.      0.     10.      0.     0.  0.      0.      0.      0.      0.      0.      0.      0.      0.  0.    0.     10.      0.      8.9995  0.      0.      0.      7.9992  0.      0.      0.      0.     10.      0.      0.      0.     -0.  0.      0.     -0.     -0.      0.      0.      0.      0.     10.  0.     -0.      0.     10.      0.    ]

bias: 15.0804, Classifacation Rate: 94.0

---2---

alpha:
[ 0.     -0.     10.      0.      0.      0.      0.      0.     10.  0.      8.7777 10.      0.      0.      0.      0.      0.      0.  0.     0.      0.      0.      0.      0.      0.      0.     10.
  8.5554  0.      0.      0.      0.      0.     10.      0.2223  0.  0.      0.     10.      0.      0.      0.      0.      0.      0.  0.      0.      0.      0.     -0.    ]

bias: 20.6334, Classifacation Rate: 94.0

1&2 Classifacation Rate: 94.0
- C100

---1---

alpha:
[  0.       0.      44.4443   0.       0.       0.     100.       0.   0.       0.       0.       0.       0.       0.       0.       0.   0.       0.       0.       0.     100.       0.      44.4443   0.   0.       0.       0.       0.       0.       0.       0.     100. 0.       0.       0.       0.       0.       0.0.       0.   0.       0.       0.       0.     100.       0.      -0.       0.  88.8886   0.    ]

bias: 11.2932, Classifacation Rate: 92.0

---2---

alpha:
[ -0.      -0.     100.      -0.      -0.      -0.      -0.      -0. 100.      -0.      48.9796  -0.      -0.      -0.      -0.      -0.  -0.      -0.      -0.      -0.      -0.      -0.      -0.      -0.  -0.      -0.      69.3878  -0.      -0.      -0.      -0.      -0.  -0.     100.      10.2041  -0.      -0.      -0.      69.3877  -0.  -0. -0. -0. -0. -0.    -0. -0. -0. -0. -0.    ] 

bias: 23.2941, Classifacation Rate: 94.0

1&2 Classifacation Rate: 93.0
## RBF SVM
### Data
- C10, sigma5

---1---

alpha:
[10.      8.9618 10.     -0.     10.     -0.     10.     -0.     -0.  0.     -0.      0.     -0.     10.     -0.     -0.      8.9618 -0.  8.9618 -0.     10.     -0.     10.     -0.     -0.     -0.     10.  0.      6.595  -0.      0.     10.      0.      0.     -0.     10. 10.     -0.     10.      0.2905 -0.     10.      0.      0.     10. -0.     10.      0.     10.     -0.    ]

bias: 0.005, Classifacation Rate: 90.0

---2---

alpha:
[10.     10.     10.     10.     -0.     -0.     -0.     -0.     10. 10.     10.     10.      2.0666 -0.     -0.      0.     10.     -0. -0.     -0.     -0.     -0.     -0.     -0.     -0.     -0.     10. 10.     -0.     -0.     -0.     -0.     -0.     10.     10.     -0. -0.     -0.     10.     -0.     -0.      2.0666 10.     -0.     -0. -0.     10.     10.     -0.     10.    ]

bias: -0.0783, Classifacation Rate: 96

1&2 Classifacation Rate: 93
- Part2: Step6
- C 1 , sigma: 1
> ---1---
> bias:  10.6272
> score:  96.0 

> ---2---
> bias:  12.4949
score:  96.0
- C 1 , sigma: 0.5
> ---1---
> bias:  9.4189
> score:  90.0 

> ---2---
> bias:  9.3192
score:  96.0
- C 1 , sigma: 0.1
> ---1---
> bias:  0.9857
> score:  94.0 

> ---2---
> bias:  0.9759
score:  96.0
- C 1 , sigma: 0.05
> ---1---
> bias:  0.4181
> score:  92.0 

> ---2---
> bias:  0.4153
score:  96.0
- C 10 , sigma: 1
> ---1---
> bias:  15.0804
> score:  94.0 

> ---2---
> bias:  20.6334
score:  94.0
- C 10 , sigma: 0.5
> ---1---
> bias:  19.4609
> score:  94.0 

> ---2---
> bias:  18.3304
score:  96.0
- C 10 , sigma: 0.1
> ---1---
> bias:  9.8574
> score:  94.0 

> ---2---
> bias:  9.7586
score:  96.0
- C 10 , sigma: 0.05
> ---1---
> bias:  4.1808
> score:  92.0 

> ---2---
> bias:  4.1534
score:  96.0
- C 100 , sigma: 1
> ---1---
> bias:  11.2932
> score:  92.0 

> ---2---
> bias:  23.2941
score:  94.0
- C 100 , sigma: 0.5
> ---1---
> bias:  22.4265
> score:  94.0 

> ---2---
> bias:  36.5018
score:  94.0
- C 100 , sigma: 0.1
> ---1---
> bias:  44.1793
> score:  94.0 

> ---2---
> bias:  44.4647
score:  96.0
- C 100 , sigma: 0.05
> ---1---
> bias:  41.4282
> score:  92.0 

> ---2---
> bias:  39.6694
score:  96.0

> c_rate:  [96. 93. 95. 94. 94. 95. 95. 94. 93. 94. 95. 94.]
## Polynomial SVM
### Data
- C10, p1

---1---

alpha: [ 0.      0.      8.9995  0.      0.      0.     10.      0.      0.  0.      0.      0.      0.      0.      0.      0.      0.      0.  0.      0.     10.      0.      8.9995  0.      0.      0.      7.9992  0.      0.      0.      0.     10.      0.      0.      0.     -0.  0.      0.     -0.     -0.      0.      0.      0.      0.     10.  0.     -0.      0.     10.      0.    ]

bias: 15.0804, score: 94.0

---2---

alpha: [ 0.     -0.     10.      0.      0.      0.      0.      0.     10.  0.      8.7777 10.      0.      0.      0.      0.      0.      0.  0.      0.      0.      0.      0.      0.      0.      0.     10.  8.5554  0.      0.      0.      0.      0.     10.      0.2223  0.  0.      0.     10.      0.      0.      0.      0.      0.      0.  0.      0.      0.      0.     -0.    ]
bias: 20.6334, score: 94

classification Rate: 94
- Part3: Step6
- first bais then Classification Rate
- C 1 , p: 2
> ---1---
> bias:  12.0977
> score:  90.0 

> ---2---
> bias:  12.4033
score:  94.0
- C 1 , p: 3
> ---1---
> bias:  10.6126
> score:  90.0 

> ---2---
> bias:  8.156
score:  94.0
- C 1 , p: 4
> ---1---
> bias:  68.974
> score:  86.0 

> ---2---
> bias:  17.2073
score:  86.0
- C 1 , p: 5
> ---1---
> bias:  2105.2213
> score:  84.0 

> ---2---
> bias:  2172.9477
score:  92.0


- C 10 , p: 2
> ---1---
> bias:  14.7218
> score:  90.0 

> ---2---
> bias:  12.1343
score:  94.0
- C 10 , p: 3
> ---1---
> bias:  8.5454
> score:  88.0 

> ---2---
> bias:  12.1464
score:  92.0
- C 10 , p: 4
> ---1---
> bias:  14.5668
> score:  86.0 

> ---2---
> bias:  -45.195
score:  14.0000
- C 10 , p: 5
> ---1---
> bias:  -427.9562
> score:  18.0 

> ---2---
> bias:  2172.9477
score:  92.0


- C 100 , p: 2
> ---1---
> bias:  14.2412
> score:  90.0 

> ---2---
> bias:  10.1673
score:  94.0
- C 100 , p: 3
> ---1---
> bias:  8.4615
> score:  88.0 

> ---2---
> bias:  21.1824
score:  86.0
- C 100 , p: 4
> ---1---
> bias:  18.0117
> score:  86.0 

> ---2---
> bias:  -45.195
score:  14.0000
- C 100 , p: 5
> ---1---
> bias:  2514.2475
> score:  84.0 

> ---2---
> bias:  2172.9477
score:  92.0

> c_rate:  [92. 92. 86. 88. 92. 90. 50. 55. 92. 87. 50. 88.]
## Discussion and results presenting
> 1.
> Linear 沒有改變映射空間，而kernel則是會將原始資料映射到更高維的空間，為使classes有更高的分離度。

> 2.
> RBF的分類率都在90%以上，但polynomial都偏低，而且幾乎都低於linear，造成這種結果的原因應該是資料的分佈造成的。
> 而且polynomial在(c100, p4)&(c10, p4)&(C10 , p4)時都有一組分類率特別低，應該是因為將該組train data升維後資料會混在一起，造成train出來的模型變得無法使用。

> 3.
> 直觀上，良好的分離是通過與任何類的最近訓練數據點具有最大距離的超平面（所謂的功能邊際）來實現的。這是因為通常餘量越大，分類器的泛化誤差越低。低泛化誤差意味著實施者不太可能經歷過度擬合。原始問題可以在有限維空間中描述，但標識集在該空間中通常不是線性可分的。因此，有人提出將原始有限維空間映射到更高維空間，可能有助於該空間中的分離。為了保持計算負荷合理，SVM方案中使用的映射簡單地通過根據核函數定義變量的輸入數據向量對的點積來減少原始空間中的變量。












