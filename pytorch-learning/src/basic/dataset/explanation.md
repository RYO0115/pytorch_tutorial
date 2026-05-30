# PyTorch Dataset チュートリアル 処理フロー解説

[このページを実施](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)

`dataset.py` に登場する Dataset・Transform・可視化の処理について、各ステップで何が起きているかを解説します。

---

## 全体の処理フロー

```
① FashionMNIST をダウンロード・読み込む
        ↓
② 各画像に Transform を適用してテンソルに変換する
        ↓
③ ランダムに 9 枚選んで 3×3 グリッドで表示する
```

---

## 1. データセットの読み込み

```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)])
)
```

### 1.1 FashionMNIST とは

28×28 ピクセルのグレースケール画像 70,000 枚からなるファッション商品データセットです。

| 分割 | 枚数 | 引数 |
|------|------|------|
| 訓練データ | 60,000 枚 | `train=True` |
| テストデータ | 10,000 枚 | `train=False` |

### 1.2 各引数の意味

| 引数 | 値 | 意味 |
|------|-----|------|
| `root` | `"data"` | データファイルの保存先ディレクトリ |
| `train` | `True` / `False` | 訓練用 or テスト用データを選択 |
| `download` | `True` | ローカルになければ自動ダウンロードする |
| `transform` | `v2.Compose(...)` | 画像を取り出すたびに自動で適用する変換処理 |

### 1.3 ダウンロードされるファイル

```
data/
└── FashionMNIST/
    └── raw/
        ├── train-images-idx3-ubyte   ← 訓練画像 (60,000枚)
        ├── train-labels-idx1-ubyte   ← 訓練ラベル
        ├── t10k-images-idx3-ubyte    ← テスト画像 (10,000枚)
        └── t10k-labels-idx1-ubyte    ← テストラベル
```

---

## 2. Transform パイプライン

`transform` に渡された処理は、データセットから画像を取り出すたびに **自動で順番に適用** されます。

```python
v2.Compose([
    v2.ToImage(),                                      # ステップ 1
    v2.ToDtype(dtype=torch.float32, scale=True)        # ステップ 2
])
```

### 2.1 v2.Compose — 変換を連結する

複数の Transform を **リストの順に** 次々と適用するラッパーです。

```
入力画像
  ↓ v2.ToImage()
  ↓ v2.ToDtype(...)
出力テンソル
```

### 2.2 ステップ 1: v2.ToImage() — 画像をテンソルに変換

PIL Image（元の形式）を PyTorch の画像テンソルに変換します。

```
変換前: PIL Image  — グレースケール 28×28、ピクセル値 0〜255 (uint8)
         ↓
変換後: Tensor     — shape (1, 28, 28)、dtype uint8
                         ^
                         チャンネル数（グレースケールなので 1）
```

PyTorch の画像テンソルは **CHW 形式**（チャンネル × 高さ × 幅）で表されます。

| 次元 | 意味 | FashionMNIST での値 |
|------|------|-------------------|
| C (Channel) | 色チャンネル数 | 1（グレースケール） |
| H (Height) | 画像の高さ（ピクセル） | 28 |
| W (Width) | 画像の幅（ピクセル） | 28 |

### 2.3 ステップ 2: v2.ToDtype(dtype=torch.float32, scale=True) — 型と値域を変換

ピクセル値の **型** と **値の範囲** を変換します。

```
変換前: dtype=uint8,   値の範囲 0〜255   (整数)
         ↓  ÷ 255
変換後: dtype=float32, 値の範囲 0.0〜1.0 (小数)
```

`scale=True` を指定することで、`÷ 255` の正規化が自動で行われます。

**なぜ 0〜1 に正規化するのか**  
ニューラルネットワークは入力値が極端に大きいと勾配が不安定になるため、スケールを揃えることで学習を安定させます。

### 2.4 Transform 適用後の最終的な shape と dtype

```
training_data[i] を呼ぶと：
  img   → Tensor, shape=(1, 28, 28), dtype=torch.float32, 値域 0.0〜1.0
  label → int, 値域 0〜9
```

---

## 3. ラベルマップ

```python
labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    ...
    9: "Ankle boot",
}
```

データセットのラベルは整数（0〜9）で格納されています。`labels_map` はその整数を人が読める文字列に変換するための辞書です。

```
label = 3  →  labels_map[3]  →  "Dress"
```

---

## 4. データセットの可視化

```python
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
```

ループが 9 回（`cols × rows = 3 × 3`）繰り返され、各回で以下の処理が行われます。

### 4.1 ランダムなインデックスを生成する

```python
sample_idx = torch.randint(len(training_data), size=(1,)).item()
```

| 処理 | 説明 |
|------|------|
| `len(training_data)` | データセットの総数（訓練データは 60,000） |
| `torch.randint(60000, size=(1,))` | 0〜59,999 の整数をランダムに 1 つ生成 → shape `(1,)` のテンソル |
| `.item()` | テンソルから Python の `int` に変換 |

```
torch.randint(60000, size=(1,))  →  tensor([42817])
                        .item()  →  42817   (Python int)
```

### 4.2 画像とラベルを取り出す

```python
img, label = training_data[sample_idx]
```

インデックスでアクセスすると、Transform 済みの画像テンソルとラベル整数が返ります。

```
training_data[42817]
  → img:   Tensor, shape=(1, 28, 28), dtype=float32
  → label: int (例: 7)
```

### 4.3 サブプロットを追加する

```python
figure.add_subplot(rows, cols, i)
```

`add_subplot(rows, cols, i)` は `rows × cols` のグリッドの **i 番目** にパネルを追加します（`i` は 1 始まり）。

```
add_subplot(3, 3, 1)  → 3×3グリッドの1番目（左上）
add_subplot(3, 3, 5)  → 3×3グリッドの5番目（中央）

┌─────┬─────┬─────┐
│  1  │  2  │  3  │
├─────┼─────┼─────┤
│  4  │  5  │  6  │
├─────┼─────┼─────┤
│  7  │  8  │  9  │
└─────┴─────┴─────┘
```

### 4.4 画像を表示する

```python
plt.imshow(img.squeeze(), cmap="gray")
```

#### img.squeeze() — 不要な次元を除去する

`imshow` はグレースケール画像に **2D テンソル `(H, W)`** を期待しますが、`img` は `(1, 28, 28)` の 3D テンソルです。  
`squeeze()` はサイズが **1 の次元を削除** して shape を揃えます。

```
img         : shape = (1, 28, 28)
img.squeeze(): shape = (28, 28)    ← チャンネル次元（サイズ 1）が消える
```

#### cmap="gray" — グレースケールで表示する

`cmap="gray"` を指定しないと、matplotlib がデフォルトのカラーマップで着色してしまいます。グレースケール画像なので `"gray"` を明示します。

---

## 処理フロー まとめ

```
datasets.FashionMNIST(...)
  ↓ ファイルが無ければダウンロード
  ↓
training_data[sample_idx]  ← インデックスでアクセス
  ↓
  [内部で Transform が自動実行される]
  PIL Image (28×28, uint8)
    → v2.ToImage()  → Tensor (1, 28, 28), uint8
    → v2.ToDtype()  → Tensor (1, 28, 28), float32, 値域 0.0〜1.0
  ↓
img (1,28,28), label (int)
  ↓
img.squeeze() → (28, 28)
  ↓
plt.imshow(..., cmap="gray")  → グレースケールで描画
```
