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

---

---

# カスタムデータセットの準備・作成・読み込み

`create_custom_dataset.py` では、自分で用意した画像ファイルと CSV を使って独自のデータセットを作る方法を実装しています。

---

## 5. 必要なファイルの準備

カスタムデータセットを使うには、あらかじめ **2 種類のファイル** を用意します。

### 5.1 ディレクトリ構造

```
プロジェクト/
├── data/
│   ├── annotations.csv       ← 画像ファイル名とラベルの対応表
│   └── images/               ← 画像ファイルを入れるディレクトリ
│       ├── tshirt1.jpg
│       ├── tshirt2.jpg
│       ├── sandal1.jpg
│       └── ankleboot1.jpg
└── create_custom_dataset.py
```

### 5.2 アノテーション CSV の形式

CSV の **1 列目に画像ファイル名**、**2 列目にラベル（整数）** を記載します。

```
tshirt1.jpg,0
tshirt2.jpg,0
pullover1.jpg,2
sandal1.jpg,5
ankleboot1.jpg,9
```

| 列 | 内容 | コード内での参照 |
|----|------|----------------|
| 0 列目（1 列目） | 画像ファイル名 | `self.img_labels.iloc[idx, 0]` |
| 1 列目（2 列目） | ラベル（整数） | `self.img_labels.iloc[idx, 1]` |

> `iloc[行番号, 列番号]` は pandas の**位置ベース**のアクセスです。列名ではなく左から何番目かで指定します。

ラベルの整数値と意味の対応は自分で決めます（例: 0=猫、1=犬、など）。

---

## 6. CustomImageDataset クラスの処理

```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        ...
    def __len__(self):
        ...
    def __getitem__(self, idx):
        ...
```

PyTorch の `Dataset` クラスを継承し、**3 つのメソッドを実装する** のがルールです。

| メソッド | 呼ばれるタイミング | 役割 |
|----------|------------------|------|
| `__init__` | インスタンス生成時（1 回だけ） | CSV の読み込みとパスの保存 |
| `__len__` | `len(dataset)` を呼んだとき | データ総数を返す |
| `__getitem__` | `dataset[i]` でアクセスしたとき | i 番目の画像とラベルを返す |

### 6.1 `__init__` — 初期化処理

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

インスタンス生成時に **1 回だけ** 実行されます。画像ファイルはここでは読みません。

```
pd.read_csv("data/annotations.csv")

   → DataFrame として self.img_labels に格納される

         0列目(ファイル名)   1列目(ラベル)
   行0:   tshirt1.jpg         0
   行1:   tshirt2.jpg         0
   行2:   sandal1.jpg         5
   行3:   ankleboot1.jpg      9
```

各引数の用途：

| 引数 | 型 | 用途 |
|------|-----|------|
| `annotations_file` | `str` | CSV ファイルのパス |
| `img_dir` | `str` | 画像が入っているディレクトリのパス |
| `transform` | callable / `None` | 画像に適用する変換（省略可） |
| `target_transform` | callable / `None` | ラベルに適用する変換（省略可） |

### 6.2 `__len__` — データ総数を返す

```python
def __len__(self):
    return len(self.img_labels)
```

`len(dataset)` と書いたときに呼ばれ、DataFrame の行数（= 画像の枚数）を返します。

```
CSV に 4 行あれば → len(dataset) → 4
```

DataLoader がシャッフルや分割をするときにもこの値を使います。

### 6.3 `__getitem__` — i 番目のデータを取り出す

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

`dataset[i]` と書いたときに呼ばれます。流れを 1 ステップずつ追います。

#### ステップ 1: 画像ファイルのパスを組み立てる

```python
img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
```

```
idx = 2 のとき:

  self.img_labels.iloc[2, 0]  →  "sandal1.jpg"   (CSV の 2 行目 0 列目)

  os.path.join("data/images", "sandal1.jpg")
    →  "data/images/sandal1.jpg"
```

`os.path.join` を使うことで、OS の違い（`/` と `\`）を吸収した安全なパス結合ができます。

#### ステップ 2: 画像をテンソルとして読み込む

```python
image = decode_image(img_path)
```

`torchvision.io.decode_image` は画像ファイル（JPEG・PNG など）を直接 **Tensor** として読み込みます。

```
"data/images/sandal1.jpg"
  → Tensor, shape=(C, H, W), dtype=uint8

  例: カラー画像 100×100px の場合 → shape=(3, 100, 100)
      グレースケール画像の場合    → shape=(1, H, W)
```

PIL Image を経由しないため変換ステップが少なく高速です。

#### ステップ 3: ラベルを取り出す

```python
label = self.img_labels.iloc[idx, 1]
```

```
idx = 2 のとき:

  self.img_labels.iloc[2, 1]  →  5   (CSV の 2 行目 1 列目)
```

#### ステップ 4: Transform を適用する（設定した場合のみ）

```python
if self.transform:
    image = self.transform(image)
if self.target_transform:
    label = self.target_transform(label)
```

`transform` が `None` でなければ画像に、`target_transform` が `None` でなければラベルに変換を適用します。

```
transform あり:
  image (uint8 Tensor) → transform → float32 Tensor（正規化済み）など

target_transform あり:
  label (int) → target_transform → one-hot ベクトルなど
```

#### ステップ 5: 画像とラベルをタプルで返す

```python
return image, label
```

```
dataset[2]  →  (Tensor(shape=C×H×W), 5)
               画像テンソル   ラベル整数
```

---

## 7. 実際の使い方

```python
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True)
])

dataset = CustomImageDataset(
    annotations_file="data/annotations.csv",
    img_dir="data/images",
    transform=transform
)

# データ総数を確認
print(len(dataset))          # → 4（CSVの行数）

# インデックスで取り出す
img, label = dataset[0]
print(img.shape)             # → torch.Size([C, H, W])
print(label)                 # → 0
```

---

## カスタムデータセット 処理フロー まとめ

```
【準備】
  annotations.csv（ファイル名, ラベル）
  data/images/（画像ファイル群）

  ↓ CustomImageDataset(...) でインスタンス生成

【__init__】
  pd.read_csv → DataFrame (self.img_labels) に全行を読み込む
  img_dir, transform を保存

  ↓ dataset[idx] でアクセス

【__getitem__】
  iloc[idx, 0] → ファイル名
  os.path.join(img_dir, ファイル名) → フルパス
  decode_image(フルパス) → Tensor (C, H, W), uint8
  iloc[idx, 1] → ラベル (int)
  transform があれば適用 → Tensor (C, H, W), float32
  return (image, label)
```
