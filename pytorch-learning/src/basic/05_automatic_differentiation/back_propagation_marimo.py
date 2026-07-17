import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # 自動微分（torch.autograd）

    ニューラルネットワークの学習は **「損失を小さくするようにパラメータを更新する」** ことが目標です。
    そのためには「各パラメータを少し変化させると損失がどう変わるか」= **勾配（gradient）** が必要です。

    `torch.autograd` はこの勾配計算を**自動で**行うモジュールです。

    ```
    [チュートリアル] https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
    [対応コード]     src/basic/05_automatic_differentiation/back_propagation.py
    ```
    """)
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    return F, FancyArrowPatch, FancyBboxPatch, mpatches, np, plt, torch


# ============================================================
# Section 1: なぜ自動微分が必要か
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1. なぜ自動微分が必要か？

    ニューラルネットワークの学習では、**損失関数 L** を最小化するよう
    重み **w** とバイアス **b** を更新します。

    $$w \\leftarrow w - \\alpha \\frac{\\partial L}{\\partial w}, \\quad b \\leftarrow b - \\alpha \\frac{\\partial L}{\\partial b}$$

    この **$\\frac{\\partial L}{\\partial w}$（勾配）** を手動で計算するのは、
    層が増えるほど非常に複雑になります。`torch.autograd` はこれを**自動**で計算してくれます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 勾配（微分）とは何か？（クリックで展開）": mo.md("""
### 微分の直感的な意味

微分 $\\frac{\\partial L}{\\partial w}$ は「**w を少し増やしたとき L がどれだけ増えるか**」の割合です。

$$\\frac{\\partial L}{\\partial w} \\approx \\frac{L(w + \\epsilon) - L(w)}{\\epsilon} \\quad (\\epsilon \\to 0)$$

| 勾配の値 | 意味 | 更新の方向 |
|:---:|:---|:---:|
| 正（+） | w を増やすと L が増える | w を**減らす** |
| 負（−） | w を増やすと L が減る | w を**増やす** |
| 0 | w を変えても L が変わらない | 更新しない |

### 連鎖律（Chain Rule）

ニューラルネットワークは複数の演算の合成関数なので、連鎖律で勾配を計算します：

$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}$$

PyTorch はこれを自動で追跡・計算します。
""")
    })
    return


# ============================================================
# Section 2: ネットワークの構築と計算グラフ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. ネットワークの構築と計算グラフ

    チュートリアルの1層ネットワークを構築します。
    入力 `x`（5次元）を受け取り、出力 `z`（3次元）を計算して損失を求めます。

    ```python
    import torch.nn.functional as F  # 損失関数・活性化関数を提供するモジュール

    x = torch.ones(5)          # 入力（勾配不要）
    y = torch.zeros(3)         # 正解ラベル（勾配不要）
    w = torch.randn(5, 3, requires_grad=True)   # 重み（勾配必要）
    b = torch.randn(3, requires_grad=True)      # バイアス（勾配必要）
    z = torch.matmul(x, w) + b                 # 線形変換
    loss = F.binary_cross_entropy_with_logits(z, y)  # 損失
    ```

    **`requires_grad=True`** を指定したテンソルだけが勾配追跡の対象になります。
    """)
    return


@app.cell
def _(F, torch):
    torch.manual_seed(42)

    x = torch.ones(5)
    y = torch.zeros(3)
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = F.binary_cross_entropy_with_logits(z, y)

    return b, loss, w, x, y, z


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 計算グラフ（Computational Graph）")
    return


@app.cell
def _(FancyBboxPatch, b, loss, plt, w, x, y, z):
    fig_dag, ax_dag = plt.subplots(figsize=(13, 5))
    ax_dag.set_xlim(0, 13)
    ax_dag.set_ylim(0, 5)
    ax_dag.axis("off")

    # ノード定義: (x_center, y_center, label, color, shape)
    _nodes = {
        "x":      (1.2, 3.8, f"x\nones(5)\nrequires_grad=False", "#d0e8ff", 1.8, 0.9),
        "w":      (1.2, 1.8, f"w\nrandn(5,3)\nrequires_grad=True", "#c8f7c5", 1.8, 0.9),
        "b":      (1.2, 0.5, f"b\nrandn(3)\nrequires_grad=True", "#c8f7c5", 1.8, 0.9),
        "y":      (1.2, 5.0 - 0.3, f"y\nzeros(3)\nrequires_grad=False", "#d0e8ff", 1.8, 0.9),
        "matmul": (4.5, 2.8, "MatMul\n(x @ w)", "#ffe0b2", 1.8, 0.8),
        "add":    (7.0, 2.5, "Add\n(+b)", "#ffe0b2", 1.6, 0.8),
        "z":      (9.2, 2.5, f"z\nshape(3)\ngrad_fn=AddmmBackward", "#e8d5f5", 2.0, 0.9),
        "bce":    (11.5, 2.5, "BCE\nLoss", "#ffe0b2", 1.6, 0.8),
        "loss":   (11.5, 1.1, f"loss\n{loss.item():.4f}\ngrad_fn=BCEWithLogits", "#ffd5d5", 1.8, 0.9),
    }

    for _key, (_cx, _cy, _label, _color, _w_box, _h_box) in _nodes.items():
        _box = FancyBboxPatch((_cx - _w_box/2, _cy - _h_box/2), _w_box, _h_box,
                              boxstyle="round,pad=0.08", facecolor=_color,
                              edgecolor="gray", lw=1.5)
        ax_dag.add_patch(_box)
        ax_dag.text(_cx, _cy, _label, ha="center", va="center", fontsize=7.5)

    # 矢印
    _arrows = [
        ("x",      "matmul", "#555"),
        ("w",      "matmul", "#555"),
        ("matmul", "add",    "#555"),
        ("b",      "add",    "#555"),
        ("add",    "z",      "#555"),
        ("z",      "bce",    "#555"),
        ("y",      "bce",    "#555"),
        ("bce",    "loss",   "#555"),
    ]
    _pos = {k: (v[0], v[1]) for k, v in _nodes.items()}
    for _src, _dst, _col in _arrows:
        _sx, _sy = _pos[_src]
        _dx, _dy = _pos[_dst]
        ax_dag.annotate("", xy=(_dx - _nodes[_dst][4]/2, _dy),
                        xytext=(_sx + _nodes[_src][4]/2, _sy),
                        arrowprops=dict(arrowstyle="->", color=_col, lw=1.5))

    # 凡例
    _legend_handles = [
        mpatches.Patch(facecolor="#c8f7c5", edgecolor="gray", label="学習パラメータ（requires_grad=True）"),
        mpatches.Patch(facecolor="#d0e8ff", edgecolor="gray", label="入力・ラベル（requires_grad=False）"),
        mpatches.Patch(facecolor="#ffe0b2", edgecolor="gray", label="演算ノード（grad_fn を持つ）"),
        mpatches.Patch(facecolor="#e8d5f5", edgecolor="gray", label="中間テンソル"),
        mpatches.Patch(facecolor="#ffd5d5", edgecolor="gray", label="損失（スカラー）"),
    ]
    ax_dag.legend(handles=_legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    plt.title("計算グラフ（Computational Graph）：順伝播の計算経路", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_dag
    return (ax_dag, fig_dag)


@app.cell(hide_code=True)
def _(b, mo, w, z):
    mo.md(f"""
    ### 各テンソルの `grad_fn`

    | テンソル | requires_grad | grad_fn | 意味 |
    |:---:|:---:|:---|:---|
    | `x` | False | None | 入力。勾配追跡なし |
    | `w` | True | None（leaf） | **学習パラメータ**。勾配追跡の起点 |
    | `b` | True | None（leaf） | **学習パラメータ**。勾配追跡の起点 |
    | `z` | True | `{z.grad_fn}` | w・b を使った演算結果。逆伝播関数を保持 |

    **`grad_fn`** は「このテンソルを生成した演算の逆伝播関数」へのポインタです。
    `loss.backward()` はこれを辿って勾配を計算します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 requires_grad の詳細：なぜ全テンソルに設定しないのか？（クリックで展開）": mo.md("""
### requires_grad=True が必要なテンソル

**学習で更新したいパラメータだけ**に設定します。

```python
w = torch.randn(5, 3, requires_grad=True)  # ✅ 更新したい
b = torch.randn(3, requires_grad=True)     # ✅ 更新したい
x = torch.ones(5)                          # ❌ 入力は更新しない
y = torch.zeros(3)                         # ❌ ラベルは更新しない
```

### requires_grad=True にしないと何が起きるか？

- 勾配が計算されない → パラメータを更新できない
- `grad_fn` が None になり、逆伝播がそこで止まる

### 既存テンソルに後から設定する方法

```python
t = torch.randn(3)
t.requires_grad_(True)   # in-place で変更（末尾の _ に注意）
```

### 勾配追跡のコスト

`requires_grad=True` のテンソルが関係する演算は、
すべて計算グラフに記録されるためメモリと計算コストが増えます。
推論時は `torch.no_grad()` で無効化するのが一般的です（後述）。
""")
    })
    return


# ============================================================
# Section 3: 勾配の計算
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. 勾配の計算：`loss.backward()`

    `loss.backward()` を呼ぶと、計算グラフを逆向きに辿り、
    各パラメータの勾配 `∂loss/∂w`, `∂loss/∂b` を計算します。
    計算結果は `.grad` 属性に蓄積されます。
    """)
    return


@app.cell
def _(b, loss, mo, w):
    loss.backward()

    mo.md(f"""
    ```python
    loss.backward()

    print(w.grad)   # ∂loss/∂w
    print(b.grad)   # ∂loss/∂b
    ```

    **`w.grad`**（shape: {w.grad.shape}）

    ```
    {w.grad}
    ```

    **`b.grad`**（shape: {b.grad.shape}）

    ```
    {b.grad}
    ```
    """)
    return


@app.cell
def _(b, np, plt, w):
    fig_grad, axes_grad = plt.subplots(1, 2, figsize=(12, 4))

    # w.grad ヒートマップ
    _ax_w = axes_grad[0]
    _wg = w.grad.numpy()
    _vmax_w = np.abs(_wg).max()
    _im_w = _ax_w.imshow(_wg, cmap="RdBu_r", vmin=-_vmax_w, vmax=_vmax_w, aspect="auto")
    fig_grad.colorbar(_im_w, ax=_ax_w, shrink=0.8)
    for _ri in range(_wg.shape[0]):
        for _ci in range(_wg.shape[1]):
            _ax_w.text(_ci, _ri, f"{_wg[_ri,_ci]:.3f}", ha="center", va="center", fontsize=9)
    _ax_w.set_title("w.grad　∂loss/∂w　shape(5,3)\n（正 → w を減らすと loss 減、負 → 増やすと loss 減）", fontsize=10)
    _ax_w.set_xlabel("出力ニューロン (0〜2)")
    _ax_w.set_ylabel("入力ニューロン (0〜4)")

    # b.grad 棒グラフ
    _ax_b = axes_grad[1]
    _bg = b.grad.numpy()
    _colors_b = ["orangered" if _v > 0 else "steelblue" for _v in _bg]
    _bars_b = _ax_b.bar(range(3), _bg, color=_colors_b, edgecolor="white")
    for _bar, _val in zip(_bars_b, _bg):
        _ax_b.text(_bar.get_x() + _bar.get_width()/2,
                   _val + (0.005 if _val >= 0 else -0.01),
                   f"{_val:.4f}", ha="center",
                   va="bottom" if _val >= 0 else "top", fontsize=10)
    _ax_b.axhline(0, color="black", lw=0.8)
    _ax_b.set_title("b.grad　∂loss/∂b　shape(3)\n（正 → b を減らすと loss 減）", fontsize=10)
    _ax_b.set_xlabel("出力ニューロン (0〜2)")
    _ax_b.set_ylabel("∂loss/∂b")
    _ax_b.grid(True, axis="y", alpha=0.3)

    plt.suptitle("backward() 後の勾配：各パラメータが loss にどう影響するか", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_grad
    return axes_grad, fig_grad


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 勾配を使ったパラメータ更新のイメージ（クリックで展開）": mo.md("""
### 勾配降下法による1ステップの更新

```python
learning_rate = 0.01

with torch.no_grad():           # 更新計算自体は勾配追跡不要
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

# 次の backward() のために勾配をリセット
w.grad.zero_()
b.grad.zero_()
```

### なぜ勾配をリセットするのか？

PyTorch は `backward()` を呼ぶたびに **勾配を累積（加算）** します。
リセットしないと次のステップの勾配に前のステップの勾配が混ざってしまいます。

```python
loss1.backward()   # w.grad = g1
loss2.backward()   # w.grad = g1 + g2  ← 意図しない累積！

w.grad.zero_()     # リセット
loss2.backward()   # w.grad = g2       ← 正しい
```

### .grad が None のテンソルに注意

`requires_grad=True` でも `backward()` を呼ぶ前は `.grad` が `None` です。
`if w.grad is not None: w.grad.zero_()` のように確認するか、
`w.grad = None` で明示的に None に戻す方法もあります。
""")
    })
    return


# ============================================================
# Section 4: 勾配追跡の無効化
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. 勾配追跡の無効化

    推論時やパラメータ更新時など、**勾配が不要な場面**では追跡を無効化して
    メモリと計算コストを削減できます。方法は2つあります。
    """)
    return


@app.cell
def _(F, mo, torch, w, x, y):
    # 方法1: torch.no_grad()
    with torch.no_grad():
        _z_no_grad = torch.matmul(x, w) + torch.zeros(3)
        _loss_no_grad = F.binary_cross_entropy_with_logits(
            _z_no_grad, y
        )

    # 方法2: .detach()
    _w_detached = w.detach()

    mo.md(f"""
    ### 方法1：`torch.no_grad()` コンテキスト

    ```python
    with torch.no_grad():
        z = torch.matmul(x, w) + b
        loss = F.binary_cross_entropy_with_logits(z, y)
    ```

    - `with` ブロック内の演算は計算グラフに**記録されない**
    - z.requires_grad → **{_z_no_grad.requires_grad}**（False になる）
    - 推論ループ全体に使うのが一般的

    ---

    ### 方法2：`.detach()`

    ```python
    w_detached = w.detach()
    ```

    - 同じデータを持つが計算グラフから**切り離された**新しいテンソルを返す
    - w.requires_grad → **{w.requires_grad}**（元のまま True）
    - w_detached.requires_grad → **{_w_detached.requires_grad}**（False）
    - 「このテンソルの値だけ使いたい（履歴は不要）」な場面で使用
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 no_grad と detach の使い分け（クリックで展開）": mo.md("""
### `torch.no_grad()` を使う場面

```python
# 1. 推論（テスト）時
model.eval()
with torch.no_grad():
    predictions = model(test_input)

# 2. パラメータ更新時
with torch.no_grad():
    w -= lr * w.grad

# 3. 検証ループ
for X, y in val_dataloader:
    with torch.no_grad():
        pred = model(X)
        val_loss = loss_fn(pred, y)
```

### `.detach()` を使う場面

```python
# 1. 途中の値をログや可視化に使いたいとき
loss_value = loss.detach().item()  # グラフを切り離してスカラーに

# 2. 転移学習でエンコーダの出力だけ使うとき
features = encoder(x).detach()  # エンコーダへの勾配を流さない
output = decoder(features)

# 3. GAN のような複雑な学習ループ
fake = generator(noise).detach()   # 生成器を固定して識別器だけ更新
d_loss = discriminator(fake)
```

### パフォーマンスへの影響

| | 勾配追跡あり | `no_grad` / `detach` |
|:---|:---:|:---:|
| メモリ | 計算グラフ分多い | 少ない |
| 速度 | 遅い | 速い（約10〜50%削減） |
| `backward()` | 使える | 使えない |
""")
    })
    return


# ============================================================
# Section 5: 計算グラフのしくみ（DAG）
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. 計算グラフのしくみ（DAG）

    PyTorch は順伝播の計算経路を**有向非巡回グラフ（DAG）**として動的に構築します。

    - **順伝播**：演算を実行しながら計算グラフを構築
    - **逆伝播**：`loss.backward()` でグラフを逆向きに辿り、連鎖律で勾配を計算
    - **グラフの再構築**：`backward()` を呼ぶとグラフは**破棄**され、次の順伝播で再構築される（動的グラフ）
    """)
    return


@app.cell
def _(FancyBboxPatch, mpatches, plt):
    fig_bp, ax_bp = plt.subplots(figsize=(13, 5.5))
    ax_bp.set_xlim(0, 13)
    ax_bp.set_ylim(0, 6)
    ax_bp.axis("off")

    # ノード定義
    _bp_nodes = {
        "x":    (1.2, 4.2, "x\n(leaf)", "#d0e8ff", 1.4, 0.8),
        "w":    (1.2, 2.5, "w\n(leaf)", "#c8f7c5", 1.4, 0.8),
        "b":    (1.2, 0.9, "b\n(leaf)", "#c8f7c5", 1.4, 0.8),
        "y":    (1.2, 5.5, "y\n(leaf)", "#d0e8ff", 1.4, 0.8),
        "mm":   (4.0, 3.3, "MatMulBackward\n(順: x@w)", "#ffe0b2", 2.0, 0.8),
        "add":  (6.8, 2.7, "AddBackward\n(順: +b)", "#ffe0b2", 1.9, 0.8),
        "bce":  (9.6, 2.7, "BCELossBackward\n(順: bce(z,y))", "#ffe0b2", 2.1, 0.8),
        "loss": (12.0, 2.7, "loss\n(scalar)", "#ffd5d5", 1.4, 0.8),
    }
    _bp_pos = {k: (v[0], v[1]) for k, v in _bp_nodes.items()}

    for _k, (_cx, _cy, _lbl, _col, _bw, _bh) in _bp_nodes.items():
        _box = FancyBboxPatch((_cx-_bw/2, _cy-_bh/2), _bw, _bh,
                              boxstyle="round,pad=0.08", facecolor=_col,
                              edgecolor="gray", lw=1.5)
        ax_bp.add_patch(_box)
        ax_bp.text(_cx, _cy, _lbl, ha="center", va="center", fontsize=8)

    # 順伝播の矢印（青）
    _fwd = [("x","mm"),("w","mm"),("mm","add"),("b","add"),("add","bce"),("y","bce"),("bce","loss")]
    for _s, _d in _fwd:
        _sx, _sy = _bp_pos[_s]
        _dx, _dy = _bp_pos[_d]
        ax_bp.annotate("", xy=(_dx - _bp_nodes[_d][4]/2, _dy),
                       xytext=(_sx + _bp_nodes[_s][4]/2, _sy),
                       arrowprops=dict(arrowstyle="->", color="steelblue", lw=2.0))

    # 逆伝播の矢印（赤、下にずらす）
    _bwd = [("loss","bce"),("bce","add"),("add","mm"),("mm","w"),("add","b")]
    for _s, _d in _bwd:
        _sx, _sy = _bp_pos[_s]
        _dx, _dy = _bp_pos[_d]
        ax_bp.annotate("", xy=(_dx + _bp_nodes[_d][4]/2, _dy - 0.35),
                       xytext=(_sx - _bp_nodes[_s][4]/2, _sy - 0.35),
                       arrowprops=dict(arrowstyle="->", color="orangered", lw=2.0,
                                       connectionstyle="arc3,rad=0.0"))

    # ラベル
    ax_bp.text(6.5, 4.2, "順伝播（Forward Pass）：計算グラフを構築", fontsize=10,
               color="steelblue", fontweight="bold")
    ax_bp.text(6.5, 0.2, "逆伝播（Backward Pass）：勾配を w, b に蓄積", fontsize=10,
               color="orangered", fontweight="bold")
    ax_bp.text(9.3, 1.5, "∂loss/∂w, ∂loss/∂b\nを計算", fontsize=9, color="orangered",
               ha="center")

    _handles = [
        mpatches.Patch(facecolor="none", edgecolor="steelblue", lw=2, label="順伝播（青矢印）"),
        mpatches.Patch(facecolor="none", edgecolor="orangered", lw=2, label="逆伝播（赤矢印）"),
    ]
    ax_bp.legend(handles=_handles, loc="upper right", fontsize=9)
    plt.title("順伝播で計算グラフを構築 → backward() で逆方向に勾配を伝播", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig_bp
    return ax_bp, fig_bp


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 勾配の累積問題と対処法（クリックで展開）": mo.md("""
### 問題：backward() を複数回呼ぶと勾配が累積する

```python
w = torch.randn(5, 3, requires_grad=True)

for step in range(3):
    z = torch.matmul(x, w) + b
    loss = F.binary_cross_entropy_with_logits(z, y)
    loss.backward()
    print(w.grad)   # step=0: g, step=1: g+g, step=2: g+g+g  ← 累積してしまう！
```

### 対処法：各ステップ前に勾配をゼロにする

```python
for step in range(3):
    # ① 勾配をリセット
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    # ② 順伝播
    z = torch.matmul(x, w) + b
    loss = F.binary_cross_entropy_with_logits(z, y)

    # ③ 逆伝播
    loss.backward()

    # ④ パラメータ更新
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
```

実際には `optimizer.zero_grad()` を使うのが一般的です（次のチュートリアルで登場）。

### なぜ累積がデフォルトなのか？

累積が便利な場面（ミニバッチの勾配を積み上げる等）があるため、
PyTorch は意図的にデフォルトを「累積」にしています。
""")
    })
    return


# ============================================================
# Section 6: ヤコビアン積
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6. ヤコビアン積（Jacobian Products）

    `loss.backward()` は「損失がスカラー」の場合のみ引数なしで呼べます。
    出力がベクトルの場合は **`backward(v)`** の形で「勾配ベクトル v」を指定します。
    これは完全なヤコビアン行列ではなく **ヤコビアン積（v · J）** を計算します。
    """)
    return


@app.cell
def _(mo, torch):
    torch.manual_seed(0)
    _inp = torch.eye(4, 5, requires_grad=True)
    _out = (_inp + 1).pow(2) / 2

    # 完全なヤコビアン積を1行ずつ計算
    _jacobian_rows = []
    for _i in range(_out.shape[0]):
        for _j in range(_out.shape[1]):
            _v = torch.zeros_like(_out)
            _v[_i, _j] = 1.0
            _out.backward(_v, retain_graph=True)
            _jacobian_rows.append(_inp.grad.clone().flatten())
            _inp.grad.zero_()

    mo.md(f"""
    ```python
    inp = torch.eye(4, 5, requires_grad=True)   # shape (4, 5)
    out = (inp + 1).pow(2) / 2                  # shape (4, 5)  ← スカラーでない！

    # 特定方向のヤコビアン積を計算する場合
    v = torch.ones_like(out)   # 方向ベクトル
    out.backward(v)            # v · J を計算
    ```

    - `out` は shape `{tuple(_out.shape)}` の行列 → スカラーではない
    - `out.backward()` だけでは **エラー**（引数が必要）
    - 代わりに方向ベクトル `v` を渡して **ヤコビアン積** を計算する

    ### なぜヤコビアン積を使うのか？

    ヤコビアン行列をそのまま計算すると shape (out の要素数 × inp の要素数) = **{_out.numel()} × {_inp.numel()}** の大行列になります。
    ニューラルネットワークでは損失が**スカラー**なので、実際には全ヤコビアンが不要です。
    連鎖律の各ステップで「ヤコビアン × 前層のベクトル」だけ計算すれば勾配が求まります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 ヤコビアン積の数学的な意味（クリックで展開）": mo.md("""
### ヤコビアン行列とは

関数 $f: \\mathbb{R}^n \\to \\mathbb{R}^m$ のヤコビアン行列 $J$ は：

$$J = \\begin{pmatrix}
\\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}{\\partial x_n} \\\\
\\vdots & \\ddots & \\vdots \\\\
\\frac{\\partial f_m}{\\partial x_1} & \\cdots & \\frac{\\partial f_m}{\\partial x_n}
\\end{pmatrix}$$

これは $m \\times n$ 行列になります。

### なぜ積（vJp）を使うのか

ニューラルネットワークの損失はスカラーなので、最終的に必要なのは：

$$\\frac{\\partial L}{\\partial \\mathbf{x}} = \\frac{\\partial L}{\\partial \\mathbf{out}} \\cdot J$$

これはヤコビアン全体を計算せず、**ベクトル × ヤコビアン** の形で計算できます。
PyTorch は内部でこの vJp（vector-Jacobian product）を効率的に計算しています。

### `retain_graph=True` について

`backward()` はデフォルトで計算グラフを破棄します。
複数回 `backward()` を呼ぶ場合（例：ヤコビアンの各行を計算）は `retain_graph=True` が必要です。

```python
for v in v_list:
    out.backward(v, retain_graph=True)  # グラフを保持
```
""")
    })
    return


# ============================================================
# Section 7: 偏微分の変化と収束条件
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. 偏微分の変化と学習の完了条件

    学習ループが進むにつれて勾配（偏微分）はどう変わり、いつ「学習完了」と判断できるのか？

    #### なぜ ∂L/∂w → 0 が「学習完了」の合図か？

    勾配降下法の更新式：

    $$w \leftarrow w - \alpha \frac{\partial L}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

    | ∂L/∂w の値 | 意味 | パラメータの動き |
    |:---:|:---|:---|
    | 大きい | w を動かすと loss が大きく変わる | 大幅に更新 |
    | 小さい（ゼロ近く） | どちらに動いても loss がほぼ変わらない | 更新量 ≈ ゼロ |
    | ゼロ | loss の極値に到達 | 更新なし = 学習完了 |

    #### 実用的な収束判定：勾配ノルム

    `w` は行列（5×3）、`b` はベクトル（3）なので、すべての偏微分を1つの指標にまとめます：

    $$\|\nabla L\| = \sqrt{\|\nabla_w L\|_F^2 + \|\nabla_b L\|^2} < \varepsilon$$

    $\|\nabla_w L\|_F$ はフロベニウスノルム（行列の全要素の二乗和の平方根）です。
    閾値 $\varepsilon$（例：$10^{-2}$）を下回ったら「収束した」とみなします。
    """)
    return


@app.cell
def _(mo):
    bp_lr = mo.ui.slider(start=0.005, stop=0.10, step=0.005, value=0.02, label="学習率 α")
    bp_steps = mo.ui.slider(start=5, stop=100, step=5, value=30, label="ステップ数")
    mo.hstack([bp_lr, bp_steps])
    return bp_lr, bp_steps


@app.cell
def _(F, bp_lr, bp_steps, np, plt, torch):
    torch.manual_seed(42)
    _x_tr = torch.ones(5)
    _y_tr = torch.zeros(3)
    _w_tr = torch.randn(5, 3, requires_grad=True)
    _b_tr = torch.randn(3, requires_grad=True)

    _lr_tr = bp_lr.value
    _n_tr = bp_steps.value
    _eps_tr = 0.01

    _loss_tr_hist, _gw_norm_hist, _gb_norm_hist, _gnorm_tr_hist = [], [], [], []

    for _ in range(_n_tr):
        # ① 順伝播
        _z_tr = torch.matmul(_x_tr, _w_tr) + _b_tr
        _loss_tr = F.binary_cross_entropy_with_logits(_z_tr, _y_tr)
        _loss_tr_hist.append(_loss_tr.item())

        # ② 逆伝播（偏微分を自動計算）
        _loss_tr.backward()

        # ③ 更新前の勾配ノルムを記録
        _gw_n = _w_tr.grad.norm().item()
        _gb_n = _b_tr.grad.norm().item()
        _gw_norm_hist.append(_gw_n)
        _gb_norm_hist.append(_gb_n)
        _gnorm_tr_hist.append(np.hypot(_gw_n, _gb_n))

        # ④ パラメータ更新・勾配リセット
        with torch.no_grad():
            _w_tr -= _lr_tr * _w_tr.grad
            _b_tr -= _lr_tr * _b_tr.grad
        _w_tr.grad.zero_()
        _b_tr.grad.zero_()

    _steps_tr = list(range(_n_tr))
    _conv_tr = next((i for i, v in enumerate(_gnorm_tr_hist) if v < _eps_tr), None)
    _annot_tr = sorted(set([0, _n_tr // 4, _n_tr // 2, 3 * _n_tr // 4, _n_tr - 1]))

    fig_conv, axes_conv = plt.subplots(2, 2, figsize=(13, 9))

    # --- 左上：損失の推移 ---
    _ax_loss = axes_conv[0, 0]
    _ax_loss.plot(_steps_tr, _loss_tr_hist, "o-", color="orangered", lw=2, ms=4)
    _ax_loss.fill_between(_steps_tr, _loss_tr_hist, min(_loss_tr_hist), alpha=0.12, color="orangered")
    if _conv_tr is not None:
        _ax_loss.axvline(_conv_tr, color="purple", lw=2, linestyle="-.", alpha=0.7)
    for _i in _annot_tr:
        _ax_loss.annotate(f"{_loss_tr_hist[_i]:.3f}", xy=(_steps_tr[_i], _loss_tr_hist[_i]),
                           xytext=(0, 8), textcoords="offset points",
                           fontsize=7.5, color="orangered", ha="center")
    _ax_loss.set_xlabel("ステップ数")
    _ax_loss.set_ylabel("損失 L（BCE）")
    _ax_loss.set_title("損失の推移\n学習が進むほど損失が減少", fontsize=10)
    _ax_loss.grid(True, alpha=0.3)

    # --- 右上：∂L/∂w ノルムと ∂L/∂b ノルム ---
    _ax_grads = axes_conv[0, 1]
    _ax_grads.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.5)
    _ax_grads.plot(_steps_tr, _gw_norm_hist, "o-", color="steelblue", lw=2, ms=4,
                    label="‖∂L/∂w‖（重み勾配ノルム）")
    _ax_grads.plot(_steps_tr, _gb_norm_hist, "s-", color="seagreen", lw=2, ms=4,
                    label="‖∂L/∂b‖（バイアス勾配ノルム）")
    _ax_grads.fill_between(_steps_tr, _gw_norm_hist, 0, alpha=0.10, color="steelblue")
    _ax_grads.fill_between(_steps_tr, _gb_norm_hist, 0, alpha=0.10, color="seagreen")
    if _conv_tr is not None:
        _ax_grads.axvline(_conv_tr, color="purple", lw=2, linestyle="-.", alpha=0.7,
                           label=f"収束 step {_conv_tr}")
    _ax_grads.set_xlabel("ステップ数")
    _ax_grads.set_ylabel("勾配ノルム")
    _ax_grads.set_title("各パラメータの勾配ノルムの変化\nゼロに近づく = 各パラメータが最適値に収束中", fontsize=10)
    _ax_grads.legend(fontsize=8)
    _ax_grads.grid(True, alpha=0.3)

    # --- 左下：合計勾配ノルム ‖∇L‖ ---
    _ax_norm = axes_conv[1, 0]
    _ax_norm.axhline(_eps_tr, color="crimson", lw=2, linestyle=":", label=f"収束閾値 ε={_eps_tr}")
    _ax_norm.plot(_steps_tr, _gnorm_tr_hist, "o-", color="orangered", lw=2, ms=4, label="‖∇L‖")
    _ax_norm.fill_between(_steps_tr, _gnorm_tr_hist, 0, alpha=0.12, color="orangered")
    if _conv_tr is not None:
        _ax_norm.axvline(_conv_tr, color="purple", lw=2, linestyle="-.",
                          label=f"収束：step {_conv_tr}（‖∇L‖ < {_eps_tr}）")
        _ax_norm.scatter([_steps_tr[_conv_tr]], [_gnorm_tr_hist[_conv_tr]],
                          color="purple", s=120, zorder=6)
    for _i in _annot_tr:
        _ax_norm.annotate(f"{_gnorm_tr_hist[_i]:.3f}", xy=(_steps_tr[_i], _gnorm_tr_hist[_i]),
                           xytext=(0, 8), textcoords="offset points",
                           fontsize=7.5, color="orangered", ha="center")
    _ax_norm.set_xlabel("ステップ数")
    _ax_norm.set_ylabel("‖∇L‖ = sqrt(‖∂L/∂w‖² + ‖∂L/∂b‖²)")
    _ax_norm.set_title(
        "勾配ノルム（全体）の推移\n"
        + (f"→ step {_conv_tr} で収束（‖∇L‖ < ε={_eps_tr}）"
           if _conv_tr is not None else f"→ {_n_tr} ステップでも未収束"),
        fontsize=10)
    _ax_norm.legend(fontsize=8)
    _ax_norm.grid(True, alpha=0.3)

    # --- 右下：数値テーブル ---
    _ax_tbl = axes_conv[1, 1]
    _ax_tbl.axis("off")
    _show_tr = sorted(set([0] + list(range(0, _n_tr, max(1, _n_tr // 7))) + [_n_tr - 1]
                           + ([] if _conv_tr is None else [_conv_tr])))
    _rows_tr = [[str(_i), f"{_loss_tr_hist[_i]:.4f}", f"{_gw_norm_hist[_i]:.4f}",
                  f"{_gb_norm_hist[_i]:.4f}", f"{_gnorm_tr_hist[_i]:.4f}"]
                 for _i in _show_tr if _i < _n_tr]
    _tbl_tr = _ax_tbl.table(
        cellText=_rows_tr,
        colLabels=["step", "損失 L", "‖∂L/∂w‖", "‖∂L/∂b‖", "‖∇L‖"],
        loc="center", cellLoc="center")
    _tbl_tr.auto_set_font_size(False)
    _tbl_tr.set_fontsize(8.5)
    _tbl_tr.scale(1.1, 1.8)
    for (_r, _c), _cell in _tbl_tr.get_celld().items():
        if _r == 0:
            _cell.set_facecolor("#2c5f8a")
            _cell.set_text_props(color="white", fontweight="bold")
        elif _conv_tr is not None and _show_tr[_r - 1] == _conv_tr:
            _cell.set_facecolor("#e8d5f5")
        elif _r % 2 == 0:
            _cell.set_facecolor("#f0f4f8")
    _ax_tbl.set_title("各ステップの詳細数値（紫行 = 収束ステップ）", fontsize=10, pad=10)

    _title_tr = (f"step {_conv_tr} で収束（‖∇L‖ = {_gnorm_tr_hist[_conv_tr]:.4f} < ε={_eps_tr}）"
                  if _conv_tr is not None
                  else f"{_n_tr} ステップ後も未収束（‖∇L‖ = {_gnorm_tr_hist[-1]:.4f} > ε={_eps_tr}）")
    plt.suptitle(f"1層ネットワークの偏微分変化と収束：{_title_tr}",
                  fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_conv
    return axes_conv, fig_conv


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **グラフの読み方**
    >
    > - **損失グラフ（左上）**：ステップが進むほど BCE 損失が減少します。
    >   「予測が正解に近づいている」ことを表します。
    >
    > - **勾配ノルムグラフ（右上）**：`‖∂L/∂w‖`（5×3 の重み行列全体）と
    >   `‖∂L/∂b‖`（3次元バイアス）がともにゼロへ向かいます。
    >   それぞれが「もうこれ以上更新しても loss が変わらない」点に近づいていることを表します。
    >
    > - **合計勾配ノルム ‖∇L‖（左下）**：w と b の勾配をまとめた「全体的な更新の勢い」です。
    >   赤い点線（ε）を下回った瞬間（紫の縦線）が **収束 = 学習完了** の判定点です。
    >
    > - **数値テーブル（右下）**：紫色の行が収束したステップです。

    #### スライダーで確認できること

    | 操作 | 観察できること |
    |:---|:---|
    | 学習率 α を大きくする（0.07 以上） | 勾配が振動・増加し、損失も不安定になる |
    | 学習率 α を小さくする（0.005） | 勾配がゆっくり減衰、収束に多くのステップが必要 |
    | ステップ数を増やす（80〜100） | ‖∇L‖ が十分ゼロに近づくまで追跡できる |

    #### PyTorch 学習ループへの組み込み方

    ```python
    for step in range(max_steps):
        z = torch.matmul(x, w) + b
        loss = F.binary_cross_entropy_with_logits(z, y)

        loss.backward()                    # 逆伝播（偏微分を自動計算）

        # 収束判定：‖∇L‖ < ε なら学習完了
        grad_norm = (w.grad.norm()**2 + b.grad.norm()**2) ** 0.5
        if grad_norm < 1e-4:
            print(f"Converged at step {step}")
            break

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    ```
    """)
    return


# ============================================================
# Section 8: まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## まとめ

    | 概念 | コード | 意味 |
    |:---|:---|:---|
    | 勾配追跡の有効化 | `requires_grad=True` | このパラメータの勾配を計算する |
    | 演算の追跡 | 自動（順伝播で記録） | 計算グラフ（DAG）を構築 |
    | 勾配の計算 | `loss.backward()` | 逆伝播で各パラメータの `.grad` に蓄積 |
    | 勾配の参照 | `w.grad` | `∂loss/∂w` の値 |
    | 勾配のリセット | `w.grad.zero_()` | 次のステップ前に必須 |
    | 追跡の無効化 | `torch.no_grad()` | 推論・更新時のコスト削減 |
    | グラフからの切り離し | `.detach()` | 値だけ使い、勾配を流さない |

    ### 学習ループの全体像

    ```
    for 各バッチ:
        ① optimizer.zero_grad()     # 勾配リセット
        ② output = model(input)     # 順伝播（計算グラフ構築）
        ③ loss = loss_fn(output, y) # 損失計算
        ④ loss.backward()           # 逆伝播（勾配計算）
        ⑤ optimizer.step()          # パラメータ更新
    ```

    この流れが `torch.autograd` を使ったニューラルネットワーク学習の基本パターンです。
    """)
    return


if __name__ == "__main__":
    app.run()
