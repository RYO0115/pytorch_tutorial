# PyTorch Tensor 演算の数学的解説

[このページを実施](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

`basic.py` に登場するベクトル・行列演算について、数学的な意味を解説します。

---

## 1. テンソルの生成

### 1.1 形状 (Shape)

テンソルの **shape** はその次元構造を表します。

| 表記 | 意味 | 例 |
|------|------|----|
| `(n,)` | n 次元ベクトル | `[1, 2, 3]` |
| `(m, n)` | m 行 n 列の行列 | 2×3 行列 |

```python
shape = (2, 3)
rand_tensor = torch.rand(shape)  # 2×3 の行列
```

$$
\text{rand\_tensor} \in \mathbb{R}^{2 \times 3}
$$

---

## 2. インデックスとスライス

### 2.1 セットアップ

`basic.py` では `torch.ones(4, 4)` で 4×4 の全要素 1 の行列を作り、一部を 0 に変更しています。

```python
tensor = torch.ones(4, 4)
tensor[:, 1] = 0   # 2列目をすべて0に
tensor[1, 2] = 0   # (1,2)要素を0に
```

結果として `tensor` の値は以下になります：

$$
A = \begin{pmatrix}
1 & 0 & 1 & 1 \\
1 & 0 & 0 & 1 \\
1 & 0 & 1 & 1 \\
1 & 0 & 1 & 1
\end{pmatrix}
$$

### 2.2 スライスの意味

| コード | 数学的意味 |
|--------|-----------|
| `tensor[0]` | 第 1 行ベクトル $\mathbf{a}_{0,:}$ |
| `tensor[:, 0]` | 第 1 列ベクトル $\mathbf{a}_{:,0}$ |
| `tensor[..., -1]` | 最終列ベクトル $\mathbf{a}_{:,n-1}$ |

---

## 3. テンソルの連結 (Concatenation)

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
```

`dim=1` は **列方向** に連結することを意味します。行列 $A \in \mathbb{R}^{4 \times 4}$ を横に 3 つ並べると：

$$
T_1 = [A \mid A \mid A] \in \mathbb{R}^{4 \times 12}
$$

`dim=0` なら行方向（縦）に連結して $\mathbb{R}^{12 \times 4}$ になります。

---

## 4. 行列積 (Matrix Multiplication)

```python
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
torch.matmul(tensor, tensor.T, out=y3)
```

`@` / `matmul` / `tensor.matmul()` はすべて同じ **行列積** を計算します。

### 4.1 行列積のルール

行列 $A \in \mathbb{R}^{m \times n}$、$B \in \mathbb{R}^{n \times p}$ に対して、積 $C = AB \in \mathbb{R}^{m \times p}$ の各要素は：

$$
C_{ij} = \sum_{k=0}^{n-1} A_{ik} \cdot B_{kj}
$$

**読み方**: 結果の $(i, j)$ 成分を求めるには、「$A$ の **第 $i$ 行** を左から右へ」と「$B$ の **第 $j$ 列** を上から下へ」を対応する位置同士で掛けてすべて足し合わせる。

```
C[i, j] を求めるとき：

  A の第 i 行 ──→  [ A[i,0]  A[i,1]  A[i,2] ... A[i,n-1] ]
                         ×        ×        ×             ×
  B の第 j 列 ──→  [ B[0,j]  B[1,j]  B[2,j] ... B[n-1,j] ]
                    ↓        ↓        ↓             ↓
                   積を全部足す  →  C[i, j]
```

### 4.2 転置 (Transpose)

```python
tensor.T
```

転置 $A^\top$ は行と列を入れ替えた行列です：

$$
(A^\top)_{ij} = A_{ji}
$$

$A \in \mathbb{R}^{m \times n}$ ならば $A^\top \in \mathbb{R}^{n \times m}$。

```
A の (0,2) 成分  →  A^T の (2,0) 成分
A の (1,3) 成分  →  A^T の (3,1) 成分
```

### 4.3 具体的な計算例 ($A A^\top$)

`basic.py` では $A \in \mathbb{R}^{4 \times 4}$ に対して $A A^\top$ を計算しています。

$$
A = \begin{pmatrix}
1 & 0 & 1 & 1 \\
1 & 0 & 0 & 1 \\
1 & 0 & 1 & 1 \\
1 & 0 & 1 & 1
\end{pmatrix}
\quad
A^\top = \begin{pmatrix}
1 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 \\
1 & 0 & 1 & 1 \\
1 & 1 & 1 & 1
\end{pmatrix}
$$

> $A^\top$ は $A$ の行と列を入れ替えたものなので、$A^\top$ の **第 $j$ 列** は $A$ の **第 $j$ 行** と同じです。

#### $C_{00}$（結果の 0 行 0 列）を求める

$A$ の **第 0 行** と $A^\top$ の **第 0 列** の要素を対応付けて掛け合わせ、足す。

```
A の第 0 行:    [ 1,  0,  1,  1 ]
                  ×   ×   ×   ×
A^T の第 0 列:  [ 1,  0,  1,  1 ]   ← A の第 0 行と同じ（転置なので）
                  ↓   ↓   ↓   ↓
積:             [ 1,  0,  1,  1 ]
                  └───┴───┴───┘
                    合計 = 3  →  C[0, 0] = 3
```

$$
C_{00} = A_{00} \cdot A^\top_{00} + A_{01} \cdot A^\top_{10} + A_{02} \cdot A^\top_{20} + A_{03} \cdot A^\top_{30}
       = 1{\cdot}1 + 0{\cdot}0 + 1{\cdot}1 + 1{\cdot}1 = 3
$$

#### $C_{01}$（結果の 0 行 1 列）を求める

$A$ の **第 0 行** と $A^\top$ の **第 1 列** を対応付ける。

```
A の第 0 行:    [ 1,  0,  1,  1 ]
                  ×   ×   ×   ×
A^T の第 1 列:  [ 1,  0,  0,  1 ]   ← A の第 1 行を転置したもの
                  ↓   ↓   ↓   ↓
積:             [ 1,  0,  0,  1 ]
                  └───┴───┴───┘
                    合計 = 2  →  C[0, 1] = 2
```

$$
C_{01} = 1{\cdot}1 + 0{\cdot}0 + 1{\cdot}0 + 1{\cdot}1 = 2
$$

#### $C_{10}$（結果の 1 行 0 列）を求める

$A$ の **第 1 行** と $A^\top$ の **第 0 列** を対応付ける。

```
A の第 1 行:    [ 1,  0,  0,  1 ]
                  ×   ×   ×   ×
A^T の第 0 列:  [ 1,  0,  1,  1 ]   ← A の第 0 行を転置したもの
                  ↓   ↓   ↓   ↓
積:             [ 1,  0,  0,  1 ]
                  └───┴───┴───┘
                    合計 = 2  →  C[1, 0] = 2
```

$$
C_{10} = 1{\cdot}1 + 0{\cdot}0 + 0{\cdot}1 + 1{\cdot}1 = 2
$$

#### 全要素の途中計算

$A$ の各行と $A^\top$ の各列を対応付けて、全16要素を求めます。

```
A の各行（固定）       A^T の各列（固定）
row0 = [1, 0, 1, 1]   col0 = [1, 0, 1, 1]   col1 = [1, 0, 0, 1]   col2 = [1, 0, 1, 1]   col3 = [1, 0, 1, 1]
row1 = [1, 0, 0, 1]
row2 = [1, 0, 1, 1]
row3 = [1, 0, 1, 1]
```

**第 0 行の計算**（$A$ の第 0 行 = $[1, 0, 1, 1]$）

```
C[0,0]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
C[0,1]:  1×1 + 0×0 + 1×0 + 1×1  =  1 + 0 + 0 + 1  =  2
C[0,2]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
C[0,3]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
```

**第 1 行の計算**（$A$ の第 1 行 = $[1, 0, 0, 1]$）

```
C[1,0]:  1×1 + 0×0 + 0×1 + 1×1  =  1 + 0 + 0 + 1  =  2
C[1,1]:  1×1 + 0×0 + 0×0 + 1×1  =  1 + 0 + 0 + 1  =  2
C[1,2]:  1×1 + 0×0 + 0×1 + 1×1  =  1 + 0 + 0 + 1  =  2
C[1,3]:  1×1 + 0×0 + 0×1 + 1×1  =  1 + 0 + 0 + 1  =  2
```

**第 2 行の計算**（$A$ の第 2 行 = $[1, 0, 1, 1]$）

```
C[2,0]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
C[2,1]:  1×1 + 0×0 + 1×0 + 1×1  =  1 + 0 + 0 + 1  =  2
C[2,2]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
C[2,3]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
```

**第 3 行の計算**（$A$ の第 3 行 = $[1, 0, 1, 1]$）

```
C[3,0]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
C[3,1]:  1×1 + 0×0 + 1×0 + 1×1  =  1 + 0 + 0 + 1  =  2
C[3,2]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
C[3,3]:  1×1 + 0×0 + 1×1 + 1×1  =  1 + 0 + 1 + 1  =  3
```

#### 結果行列

$$
C = AA^\top = \begin{pmatrix}
3 & 2 & 3 & 3 \\
2 & 2 & 2 & 2 \\
3 & 2 & 3 & 3 \\
3 & 2 & 3 & 3
\end{pmatrix}
$$

> **性質**: $A A^\top$ は常に **対称行列** ($C = C^\top$) になります。$C_{01} = C_{10} = 2$ がその例です。

---

## 5. 要素積 (Element-wise Product / Hadamard Product)

```python
z1 = tensor * tensor
z2 = tensor.mul(tensor)
torch.mul(tensor, tensor, out=z3)
```

これら 3 つはすべて同じ **アダマール積** を計算します。行列積と違い、**位置 $(i,j)$ の要素は、同じ位置 $(i,j)$ の要素とだけ掛け合わさり、結果も同じ位置 $(i,j)$ に書かれます**。

$$
Z_{ij} = A_{ij} \cdot B_{ij}
\quad \text{（結果 Z の } (i,j) \text{ 成分 = A の } (i,j) \text{ 成分 × B の } (i,j) \text{ 成分）}
$$

入力と出力の shape は同じです（$\mathbb{R}^{m \times n} \odot \mathbb{R}^{m \times n} = \mathbb{R}^{m \times n}$）。

### 5.1 全要素の途中計算

`tensor * tensor`（つまり $A \odot A$）での全16要素の流れ。  
各行について「左の $A$ の値 × 右の $A$ の値 = 結果の値」を位置ごとに示します。

**入力行列（左右ともに $A$）**

$$
A = \begin{pmatrix}
1 & 0 & 1 & 1 \\
1 & 0 & 0 & 1 \\
1 & 0 & 1 & 1 \\
1 & 0 & 1 & 1
\end{pmatrix}
$$

**第 0 行の要素積**

```
位置 (0,0):  A[0,0] × A[0,0]  =  1 × 1  =  1   →  Z[0,0] = 1
位置 (0,1):  A[0,1] × A[0,1]  =  0 × 0  =  0   →  Z[0,1] = 0
位置 (0,2):  A[0,2] × A[0,2]  =  1 × 1  =  1   →  Z[0,2] = 1
位置 (0,3):  A[0,3] × A[0,3]  =  1 × 1  =  1   →  Z[0,3] = 1
```

**第 1 行の要素積**

```
位置 (1,0):  A[1,0] × A[1,0]  =  1 × 1  =  1   →  Z[1,0] = 1
位置 (1,1):  A[1,1] × A[1,1]  =  0 × 0  =  0   →  Z[1,1] = 0
位置 (1,2):  A[1,2] × A[1,2]  =  0 × 0  =  0   →  Z[1,2] = 0
位置 (1,3):  A[1,3] × A[1,3]  =  1 × 1  =  1   →  Z[1,3] = 1
```

**第 2 行の要素積**

```
位置 (2,0):  A[2,0] × A[2,0]  =  1 × 1  =  1   →  Z[2,0] = 1
位置 (2,1):  A[2,1] × A[2,1]  =  0 × 0  =  0   →  Z[2,1] = 0
位置 (2,2):  A[2,2] × A[2,2]  =  1 × 1  =  1   →  Z[2,2] = 1
位置 (2,3):  A[2,3] × A[2,3]  =  1 × 1  =  1   →  Z[2,3] = 1
```

**第 3 行の要素積**

```
位置 (3,0):  A[3,0] × A[3,0]  =  1 × 1  =  1   →  Z[3,0] = 1
位置 (3,1):  A[3,1] × A[3,1]  =  0 × 0  =  0   →  Z[3,1] = 0
位置 (3,2):  A[3,2] × A[3,2]  =  1 × 1  =  1   →  Z[3,2] = 1
位置 (3,3):  A[3,3] × A[3,3]  =  1 × 1  =  1   →  Z[3,3] = 1
```

**結果行列**

$$
Z = A \odot A = \begin{pmatrix}
1 & 0 & 1 & 1 \\
1 & 0 & 0 & 1 \\
1 & 0 & 1 & 1 \\
1 & 0 & 1 & 1
\end{pmatrix}
$$

`tensor * tensor` は $A \odot A$ なので、各要素の自乗と等価です（$0^2=0$、$1^2=1$ なので今回は $A$ と同じ値になります）。

$$
Z_{ij} = A_{ij}^2
$$

### 5.2 行列積との比較

| 演算 | 記号 | $Z_{ij}$ の計算に使う要素 | 出力 shape |
|------|------|--------------------------|-----------|
| 行列積 | `@` / `matmul` | $A$ の **第 $i$ 行全体** と $B$ の **第 $j$ 列全体** を対応付けて掛け合わせ足す | $(m, p)$ |
| 要素積 | `*` / `mul` | $A$ の **$(i,j)$ 成分** と $B$ の **$(i,j)$ 成分** だけ | $(m, n)$ ← 入力と同じ |

---

## 6. べき乗 (Element-wise Power)

```python
tensor.pow(2)
```

各要素を $n$ 乗します：

$$
(\text{pow}(A, n))_{ij} = A_{ij}^n
$$

`tensor.pow(2)` と `tensor * tensor`（要素積）は **数値的に同じ結果** になります：

$$
A^{\odot 2} = A \odot A
$$

---

## まとめ

| コード | 数学記法 | 演算の種類 |
|--------|---------|-----------|
| `tensor @ tensor.T` | $A A^\top$ | 行列積 |
| `tensor.matmul(tensor.T)` | $A A^\top$ | 行列積（同上） |
| `tensor * tensor` | $A \odot A$ | 要素積（アダマール積） |
| `tensor.mul(tensor)` | $A \odot A$ | 要素積（同上） |
| `tensor.pow(2)` | $A^{\odot 2}$ | 要素ごとのべき乗 |
| `tensor[:, 1]` | $\mathbf{a}_{:,1}$ | 列ベクトルの抽出 |
| `tensor.T` | $A^\top$ | 転置 |

---

## 7. 偏微分（Partial Derivative）

> **DL を理解するうえで最も重要な数学概念のひとつ。**
> パラメータの「更新方向」を決めるためにどうしても必要になります。

---

### 7.1 偏微分とは何か

複数の変数を持つ関数で、**注目する変数以外をすべて定数とみなして微分する**操作です。

$$\frac{\partial f}{\partial x} \quad \text{（「x で偏微分する」と読む）}$$

#### 具体例

$$f(w, b) = w^2 + w \cdot b + b^2$$

$w$ で偏微分する（$b$ を定数扱い）：

$$\frac{\partial f}{\partial w} = 2w + b$$

$b$ で偏微分する（$w$ を定数扱い）：

$$\frac{\partial f}{\partial b} = w + 2b$$

#### 偏微分の直感的な意味

> **「$w$ を少しだけ増やすと $f$ はどれだけ変わるか？」の割合**

$$\frac{\partial f}{\partial w} \approx \frac{f(w + \varepsilon, b) - f(w, b)}{\varepsilon} \quad (\varepsilon \to 0)$$

| $\frac{\partial f}{\partial w}$ の値 | 意味 |
|:---:|:---|
| 正（$+$） | $w$ を増やすと $f$ が増える |
| 負（$-$） | $w$ を増やすと $f$ が減る |
| $0$ | $w$ を変えても $f$ は変わらない |

---

### 7.2 DL で偏微分が必要な理由

ニューラルネットワークの学習は **「損失関数 $L$ を最小化すること」** です。

```
損失関数 L は重み w とバイアス b の関数：

L = L(w_{00}, w_{01}, ..., w_{54}, b_0, b_1, b_2)
        ↑
    back_propagation.py の例では
    w は shape(5,3) → 15個の変数
    b は shape(3)   →  3個の変数
    合計18個の変数を持つ関数
```

**どのパラメータをどの方向に・どれだけ動かすと $L$ が小さくなるか**、
これを知るためには「各パラメータに対する $L$ の偏微分」が必要です。

$$\frac{\partial L}{\partial w_{00}}, \quad \frac{\partial L}{\partial w_{01}}, \quad \ldots, \quad \frac{\partial L}{\partial b_2}$$

---

### 7.3 DL での具体的な使われ方：勾配降下法

偏微分を使ってパラメータを更新するアルゴリズムを **勾配降下法** といいます。

$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

- $\alpha$：学習率（learning rate）。更新の大きさを調整する定数
- 偏微分が**正** → $w$ を増やすと $L$ が増える → $w$ を**減らす**（$-$ をかける）
- 偏微分が**負** → $w$ を増やすと $L$ が減る → $w$ を**増やす**（$-$ をかける）

これを繰り返すことで損失が小さくなる方向へパラメータが更新されていきます。

```
初期値 w （ランダム）
  ↓ 勾配 ∂L/∂w を計算
  ↓ w ← w - α * ∂L/∂w　（小さくなる方向へ移動）
  ↓ また ∂L/∂w を計算
  ↓ w ← w - α * ∂L/∂w
  ↓ ...繰り返す
最小値付近の w に収束
```

---

### 7.4 back_propagation.py の例での計算

`back_propagation.py` の1層ネットワーク：

```python
x = torch.ones(5)              # 入力
y = torch.zeros(3)             # 正解ラベル
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b    # z = xW + b
loss = BCE(z, y)               # 損失
```

必要な偏微分は：

$$\frac{\partial L}{\partial w_{ij}} \quad (i=0..4,\ j=0..2) \quad \text{と} \quad \frac{\partial L}{\partial b_j} \quad (j=0..2)$$

連鎖律（Chain Rule）を使って分解します：

$$\frac{\partial L}{\partial w_{ij}} = \underbrace{\frac{\partial L}{\partial z_j}}_{\text{BCE の微分}} \cdot \underbrace{\frac{\partial z_j}{\partial w_{ij}}}_{= x_i}$$

$$\frac{\partial L}{\partial b_j} = \underbrace{\frac{\partial L}{\partial z_j}}_{\text{BCE の微分}} \cdot \underbrace{\frac{\partial z_j}{\partial b_j}}_{= 1}$$

> **連鎖律**：$L = f(z)$、$z = g(w)$ のとき $\dfrac{\partial L}{\partial w} = \dfrac{\partial L}{\partial z} \cdot \dfrac{\partial z}{\partial w}$

PyTorch はこの計算を `loss.backward()` で自動的に行い、結果を `.grad` に格納します。

```python
loss.backward()

print(w.grad)   # ∂L/∂w  shape(5,3)
print(b.grad)   # ∂L/∂b  shape(3)
```

---

### 7.5 勾配ベクトル（Gradient Vector）

複数の偏微分をまとめたものを **勾配ベクトル**（または単に勾配）といいます。

$$\nabla_w L = \begin{pmatrix} \frac{\partial L}{\partial w_{00}} & \frac{\partial L}{\partial w_{01}} & \frac{\partial L}{\partial w_{02}} \\ \vdots & \vdots & \vdots \\ \frac{\partial L}{\partial w_{40}} & \frac{\partial L}{\partial w_{41}} & \frac{\partial L}{\partial w_{42}} \end{pmatrix} \in \mathbb{R}^{5 \times 3}$$

これがそのまま PyTorch の `w.grad`（shape(5,3)）に対応します。

| PyTorch | 数学 | 意味 |
|:---|:---|:---|
| `w.grad` | $\nabla_w L$ | $w$ の各要素に対する $L$ の偏微分行列 |
| `b.grad` | $\nabla_b L$ | $b$ の各要素に対する $L$ の偏微分ベクトル |
| `loss.backward()` | 連鎖律による $\nabla L$ の計算 | 計算グラフを逆向きに辿って全偏微分を計算 |

---

### 7.6 DL の学習ループにおける偏微分の位置づけ

```
① 順伝播：入力 → ネットワーク → 損失 L を計算
           （計算グラフを構築）

② 逆伝播：loss.backward()
           ↓ 連鎖律で ∂L/∂w, ∂L/∂b を自動計算（偏微分）

③ パラメータ更新：勾配降下法
           w ← w - α * ∂L/∂w
           b ← b - α * ∂L/∂b

④ 勾配リセット：w.grad.zero_()
                次のステップのために勾配をクリア

⑤ ① に戻る
```

**偏微分は「どこに向かってパラメータを動かすか」の羅針盤です。**
PyTorch の `autograd` はこの羅針盤を自動で計算してくれる仕組みです。
