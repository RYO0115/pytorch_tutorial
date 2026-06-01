import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # NeuralNetwork の処理フロー可視化

    `build_nn_network.py` のニューラルネットワークが、入力データをどのように変換して
    **Predicted Class** を導くかを、各ステップのデータで確認します。

    ```
    入力(1,28,28) → Flatten(1,784) → Linear→ReLU(1,512) → Linear→ReLU(1,512) → Linear(1,10) → Softmax → argmax
    ```

    入力データは **MNIST**（手書き数字 0〜9、28×28グレースケール）のテスト画像を使用します。
    モデルはランダム重みのため予測精度は低いですが、各ステップのデータ変換を観察できます。
    """)
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    return datasets, nn, np, plt, torch, transforms


@app.cell
def _(nn, torch):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    torch.manual_seed(42)
    model = NeuralNetwork()
    model.eval()
    return NeuralNetwork, model


@app.cell
def _(datasets, transforms):
    _transform = transforms.ToTensor()
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fashion_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=_transform
    )
    return class_names, fashion_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## ステップ0：入力データを選択")
    return


@app.cell
def _(mo):
    idx_slider = mo.ui.slider(
        start=0, stop=9999, step=1, value=0,
        label="MNIST テストデータのインデックス（0〜9999）",
        full_width=True,
    )
    idx_slider
    return (idx_slider,)


@app.cell
def _(class_names, fashion_dataset, idx_slider, model, torch):
    _idx = idx_slider.value
    _img, true_label = fashion_dataset[_idx]   # _img: (1, 28, 28)

    X = _img   # shape (1, 28, 28)  ← モデルは Flatten で (1, 784) に変換

    with torch.no_grad():
        x_flat   = model.flatten(X)
        x_lin1   = model.linear_relu_stack[0](x_flat)
        x_relu1  = model.linear_relu_stack[1](x_lin1)
        x_lin2   = model.linear_relu_stack[2](x_relu1)
        x_relu2  = model.linear_relu_stack[3](x_lin2)
        x_logits = model.linear_relu_stack[4](x_relu2)
        x_prob   = torch.softmax(x_logits, dim=1)
        y_pred   = x_prob.argmax(1).item()

    true_label_name = class_names[true_label]
    pred_label_name = class_names[y_pred]

    return (
        X, class_names, pred_label_name, true_label, true_label_name,
        x_flat, x_lin1, x_lin2, x_logits, x_prob, x_relu1, x_relu2, y_pred,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ1：入力画像 → Flatten

    **入力 shape**: `(1, 28, 28)` → **Flatten後 shape**: `(1, 784)`

    28×28ピクセルのグレースケール画像を、784個の数値が並ぶ1次元ベクトルに展開します。
    データの値は変わらず、**並べ方だけ変わります**。
    """)
    return


@app.cell
def _(X, np, plt, pred_label_name, true_label_name, x_flat, y_pred):
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4),
                                gridspec_kw={"width_ratios": [1, 2]})

    # 左：28×28 ヒートマップ（実際のFashionMNIST画像）
    ax_img = axes1[0]
    im_img = ax_img.imshow(X.squeeze().numpy(), cmap="gray_r", vmin=0, vmax=1)
    fig1.colorbar(im_img, ax=ax_img, shrink=0.8)
    ax_img.set_title(
        f"入力画像　shape: (1, 28, 28)\n"
        f"✅ 正解ラベル：{true_label_name}",
        fontsize=11,
    )
    ax_img.set_xlabel("列 (0〜27)")
    ax_img.set_ylabel("行 (0〜27)")

    # 右：Flatten後を1×784の横ストリップで表示
    ax_flat = axes1[1]
    flat_np = x_flat.squeeze().numpy()
    im_flat = ax_flat.imshow(flat_np.reshape(1, -1), cmap="gray_r",
                              vmin=0, vmax=1, aspect="auto")
    fig1.colorbar(im_flat, ax=ax_flat, shrink=0.8)
    ax_flat.set_title(
        f"Flatten後　shape: (1, 784)\n"
        f"min={flat_np.min():.3f}  max={flat_np.max():.3f}  （値は同じ・形だけ変わる）",
        fontsize=11,
    )
    ax_flat.set_xlabel("ピクセルインデックス (0〜783)")
    ax_flat.set_yticks([])
    ax_flat.axvline(27, color="red", lw=1, linestyle="--", alpha=0.5, label="行0の終端(27)")
    ax_flat.axvline(28, color="orange", lw=1, linestyle="--", alpha=0.5, label="行1の先頭(28)")
    ax_flat.legend(fontsize=8, loc="upper right")

    _correct = true_label_name == pred_label_name
    _result_str = f"{'✅ 正解' if _correct else '❌ 不正解'}　予測：{pred_label_name}"
    plt.suptitle(
        f"STEP 1：FashionMNIST 画像 → Flatten　｜　{_result_str}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig1
    return ax_flat, ax_img, axes1, fig1, flat_np, im_flat, im_img


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ1→2：Linear(784→512) の仕組み

    Linear層は **重み行列 W（512×784）** と **バイアス b（512）** を持ちます。

    出力の各ニューロン $j$ は、入力784個との**内積 + バイアス**で計算されます：

    $$h_j = \\underbrace{\\sum_{i=0}^{783} w_{ji} \\cdot x_i}_{\\text{内積}} + b_j$$

    - $w_{ji}$：ニューロン $j$ が入力 $i$ に対して持つ「重要度の重み」
    - 正の重み → その入力が大きいほど出力が大きくなる
    - 負の重み → その入力が大きいほど出力が小さくなる

    **重み行列を 28×28 にreshapeすると、各ニューロンが「画像のどの位置」を重視しているかが見えます。**
    """)
    return


@app.cell
def _(mo):
    neuron_slider = mo.ui.slider(
        start=0, stop=511, step=1, value=0,
        label="調べるニューロン番号（0〜511）",
        full_width=True,
    )
    neuron_slider
    return (neuron_slider,)


@app.cell
def _(model, np, plt):
    # 最初の8ニューロンの重みパターンを28×28で表示
    _W = model.linear_relu_stack[0].weight.detach().numpy()  # (512, 784)
    _vmax_w = np.abs(_W[:8]).max()

    fig_w8, _axes_w8 = plt.subplots(2, 4, figsize=(13, 6))
    for _ni in range(8):
        _ax = _axes_w8[_ni // 4, _ni % 4]
        _im = _ax.imshow(_W[_ni].reshape(28, 28), cmap="RdBu_r",
                         vmin=-_vmax_w, vmax=_vmax_w)
        fig_w8.colorbar(_im, ax=_ax, shrink=0.6)
        _ax.set_title(f"ニューロン{_ni}の重み\n(28×28にreshape)", fontsize=9)
        _ax.axis("off")

    plt.suptitle(
        "重み行列 W の最初の8行（各ニューロンが「画像のどのパターン」に反応するかを表す）",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_w8
    return (fig_w8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ### ❓ 疑問：重み W が (512, 784) ということは、28×28 の重み行列が 512 個できる？

    **→ はい、正確な理解です。**

    ```
    W の shape: (512, 784)
               ↑      ↑
          ニューロン数  入力の次元数 (= 28×28)
    ```

    各行が「1つのニューロンの重みベクトル（784次元）」で、それを 28×28 に reshape すると
    「そのニューロンが画像のどこに反応するか」を表すパターンが得られます。

    ```
    W[0]   → reshape(28, 28) → ニューロン0 の重みパターン
    W[1]   → reshape(28, 28) → ニューロン1 の重みパターン
    W[2]   → reshape(28, 28) → ニューロン2 の重みパターン
    ...
    W[511] → reshape(28, 28) → ニューロン511 の重みパターン
    ```

    ### ✅ 理解のまとめ

    > **W[512, 784] は「512 種類の画像フィルタ」と見なせる。**
    > 各フィルタが入力画像と内積を取って、そのフィルタへの「反応度」を
    > 1 つのスカラーとして出力する。それが 512 個まとまって
    > Linear 層の出力（512 次元）になる。

    上のグラフはその 512 個のうち最初の 8 個を可視化したものです。
    スライダーで 0〜511 のニューロンを選ぶと、任意のフィルタのパターンを確認できます。
    """)
    return


@app.cell
def _(X, model, neuron_slider, np, plt, x_flat, x_lin1):
    _j = neuron_slider.value
    _W = model.linear_relu_stack[0].weight.detach().numpy()  # (512, 784)
    _b = model.linear_relu_stack[0].bias.detach().numpy()    # (512,)

    _w_j   = _W[_j]                          # (784,) このニューロンの重みベクトル
    _x_np  = x_flat.squeeze().numpy()        # (784,) 入力ベクトル
    _elem  = _w_j * _x_np                    # (784,) 要素ごとの積
    _dot   = _elem.sum()                     # スカラー：内積
    _h_j   = _dot + _b[_j]                  # スカラー：バイアスを加えた出力値

    # 全512ニューロンの出力絶対値の最大値 → y軸を固定するための共通スケール
    _ymax = float(np.abs(x_lin1.squeeze().numpy()).max()) * 1.25

    fig_neuron, _axes_n = plt.subplots(1, 4, figsize=(15, 4),
                                        gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})

    _vmax_x = 1.0
    _vmax_w_j = np.abs(_w_j).max()
    _vmax_e = np.abs(_elem).max()

    # Panel 1: 入力X
    _ax0 = _axes_n[0]
    _im0 = _ax0.imshow(X.squeeze().numpy(), cmap="Blues", vmin=0, vmax=1)
    fig_neuron.colorbar(_im0, ax=_ax0, shrink=0.7)
    _ax0.set_title(f"入力 X\n(28×28)", fontsize=10)
    _ax0.axis("off")

    # Panel 2: 重みW[j]
    _ax1 = _axes_n[1]
    _im1 = _ax1.imshow(_w_j.reshape(28, 28), cmap="RdBu_r",
                       vmin=-_vmax_w_j, vmax=_vmax_w_j)
    fig_neuron.colorbar(_im1, ax=_ax1, shrink=0.7)
    _ax1.set_title(f"重み W[{_j}]\n(28×28にreshape)", fontsize=10)
    _ax1.axis("off")

    # Panel 3: 要素積 X * W[j]
    _ax2 = _axes_n[2]
    _im2 = _ax2.imshow(_elem.reshape(28, 28), cmap="RdBu_r",
                       vmin=-_vmax_e, vmax=_vmax_e)
    fig_neuron.colorbar(_im2, ax=_ax2, shrink=0.7)
    _ax2.set_title(f"要素積 X × W[{_j}]\n(各ピクセルの寄与)", fontsize=10)
    _ax2.axis("off")

    # Panel 4: 集計（棒グラフ）
    _ax3 = _axes_n[3]
    _bar_vals  = [_dot, _b[_j], _h_j]
    _bar_cols  = [
        "steelblue" if _dot >= 0 else "orangered",
        "green" if _b[_j] >= 0 else "tomato",
        "purple",
    ]
    _bar_lbls  = ["内積\n(Σ xᵢwᵢ)", f"バイアス\nb[{_j}]", f"出力\nh[{_j}]"]
    _bars_n    = _ax3.bar(_bar_lbls, _bar_vals, color=_bar_cols, edgecolor="white")
    _ax3.axhline(0, color="black", lw=0.8)
    _ax3.set_title("計算結果の集計", fontsize=10)
    _ax3.set_ylabel("値")
    _ax3.set_ylim(-_ymax, _ymax)  # 全ニューロン共通の固定y軸
    _ax3.grid(True, axis="y", alpha=0.3)
    for _bar_item, _val in zip(_bars_n, _bar_vals):
        _ax3.text(_bar_item.get_x() + _bar_item.get_width() / 2,
                  _val + (abs(_val) * 0.05 + 0.01) * (1 if _val >= 0 else -1),
                  f"{_val:.3f}", ha="center", va="bottom" if _val >= 0 else "top",
                  fontsize=9, fontweight="bold")

    plt.suptitle(
        f"ニューロン {_j} の計算：内積(X·W[{_j}]) + b[{_j}] = {_h_j:.4f}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_neuron
    return (fig_neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **スライダーで色々なニューロンを見てみよう**
    > - 重み W[j] のパターンがニューロンごとに異なる → それぞれ「異なるパターンを検出する検出器」
    > - 要素積のパターンを見ると「入力画像のどの部分にそのニューロンが反応したか」がわかる
    > - 内積の符号で「このニューロンが今の入力に対して反応した（正）か、反応しなかった（負）か」がわかる
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ2：Linear(784→512) → ReLU

    **全結合層**：784個の入力それぞれに重みを掛けて足し合わせ、512個の出力を作ります。

    $$h_j = \\sum_{i=0}^{783} w_{ji} x_i + b_j \\quad (j = 0, 1, ..., 511)$$

    **ReLU**：負の値を0に置き換えます。 $\\text{ReLU}(h) = \\max(0, h)$
    """)
    return


@app.cell
def _(np, plt, x_lin1, x_relu1):
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 7),
                                gridspec_kw={"width_ratios": [2, 1]})

    lin1_np   = x_lin1.squeeze().numpy()
    relu1_np  = x_relu1.squeeze().numpy()
    vmax_l1   = np.abs(lin1_np).max()

    # --- Linear1後 ヒートマップ ---
    ax_l1h = axes2[0, 0]
    im_l1 = ax_l1h.imshow(lin1_np.reshape(32, 16), cmap="RdBu_r",
                           vmin=-vmax_l1, vmax=vmax_l1, aspect="auto")
    fig2.colorbar(im_l1, ax=ax_l1h, shrink=0.8)
    ax_l1h.set_title("Linear(784→512) 後\nshape: (1, 512)  ← 正負両方の値が混在", fontsize=10)
    ax_l1h.set_xlabel("列インデックス (0〜15)")
    ax_l1h.set_ylabel("行インデックス (0〜31)")

    # --- Linear1後 ヒストグラム ---
    ax_l1d = axes2[0, 1]
    ax_l1d.hist(lin1_np, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax_l1d.axvline(0, color="red", lw=1.5, linestyle="--", label="x=0")
    ax_l1d.set_title("値の分布\n（負の値も多い）", fontsize=10)
    ax_l1d.set_xlabel("活性化値")
    ax_l1d.set_ylabel("個数")
    ax_l1d.legend(fontsize=9)
    ax_l1d.grid(True, alpha=0.3)

    n_neg = int((lin1_np < 0).sum())
    ax_l1d.text(0.98, 0.95, f"負の値: {n_neg}/512\n正の値: {512-n_neg}/512",
                transform=ax_l1d.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- ReLU1後 ヒートマップ ---
    ax_r1h = axes2[1, 0]
    im_r1 = ax_r1h.imshow(relu1_np.reshape(32, 16), cmap="Oranges",
                           vmin=0, vmax=vmax_l1, aspect="auto")
    fig2.colorbar(im_r1, ax=ax_r1h, shrink=0.8)
    n_zero = int((relu1_np == 0).sum())
    ax_r1h.set_title(f"ReLU 後\nshape: (1, 512)  ← 負→0（死んだニューロン: {n_zero}/512個）", fontsize=10)
    ax_r1h.set_xlabel("列インデックス (0〜15)")
    ax_r1h.set_ylabel("行インデックス (0〜31)")

    # --- ReLU1後 ヒストグラム ---
    ax_r1d = axes2[1, 1]
    pos_vals = relu1_np[relu1_np > 0]
    ax_r1d.hist(pos_vals, bins=40, color="orangered", edgecolor="white", alpha=0.8, label="正（活性）")
    ax_r1d.bar(0, n_zero, width=0.3, color="lightgray", edgecolor="gray", label=f"0（不活性: {n_zero}個）")
    ax_r1d.set_title("値の分布\n（負がすべて0に）", fontsize=10)
    ax_r1d.set_xlabel("活性化値")
    ax_r1d.set_ylabel("個数")
    ax_r1d.legend(fontsize=9)
    ax_r1d.grid(True, alpha=0.3)

    plt.suptitle("STEP 2：Linear(784→512) + ReLU の変換", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig2
    return (
        ax_l1d, ax_l1h, ax_r1d, ax_r1h,
        axes2, fig2, im_l1, im_r1,
        lin1_np, n_neg, n_zero, pos_vals,
        relu1_np, vmax_l1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ2→3：ReLUの後にまたLinearが来る理由

    ### ReLU①の出力が、そのままLinear②の入力になります

    ```
    ReLU①出力 (1, 512)  ← 0以上の値のみ
      ↓
    Linear②：W₂(512×512) との内積 + b₂   ← ここで「特徴の組み合わせ」を再学習
      ↓
    Linear②出力 (1, 512)  ← また正負混在
      ↓
    ReLU②
    ```

    ### なぜ「ReLUの後にまたLinear」が必要なのか？

    **ReLUがなければ、何層重ねても1層と同じになってしまいます：**

    $$\\text{Linear}_2(\\text{Linear}_1(\\mathbf{x})) = W_2(W_1\\mathbf{x} + b_1) + b_2 = \\underbrace{(W_2 W_1)}_{W_3}\\mathbf{x} + \\text{const}$$

    → どれだけ積み重ねても「1枚の行列との掛け算」に変わらない。

    **ReLUを挟むことで、本当の意味での「深さ」が生まれます：**

    $$\\text{Linear}_2(\\underbrace{\\text{ReLU}(\\text{Linear}_1(\\mathbf{x}))}_{\\text{ここが非線形}}) \\neq W_3 \\mathbf{x}$$

    | 層 | 何を学習するか |
    |:---|:---|
    | Linear①→ReLU① | ピクセルの基本パターン（エッジ・明暗）を検出 |
    | Linear②→ReLU② | ①の検出結果の**組み合わせ**（角・曲線など）を検出 |
    | Linear③ | ②の組み合わせから**10クラスのどれか**を判定 |
    """)
    return


@app.cell(hide_code=True)
def _(mo, model, np, x_lin2, x_relu1):
    _W2 = model.linear_relu_stack[2].weight.detach().numpy()  # (512, 512)
    _b2 = model.linear_relu_stack[2].bias.detach().numpy()    # (512,)
    _r1 = x_relu1.squeeze().numpy()   # (512,) ← ReLU①の出力
    _n_active = int((_r1 > 0).sum())

    # ニューロン0の計算を具体例として使う
    _dot2_0 = float((_W2[0] * _r1).sum())
    _h2_0   = float(_dot2_0 + _b2[0])

    # 実際の値から「活性」と「不活性」の代表インデックスを3つずつ取得
    _act_idx  = np.where(_r1 > 0)[0][:3]
    _zero_idx = np.where(_r1 == 0)[0][:3]

    _act_rows = "\n".join(
        f"| {_i} | {_r1[_i]:.4f} | {_W2[0][_i]:.4f} | **{_r1[_i] * _W2[0][_i]:.4f}** |"
        for _i in _act_idx
    )
    _zero_rows = "\n".join(
        f"| {_i} | 0.0000 | {_W2[0][_i]:.4f} | **0.0000**（ゼロ確定） |"
        for _i in _zero_idx
    )

    _detail = mo.md(f"""
### ReLU① → Linear② の計算式

Linear②のニューロン $j$ の出力は、ReLU①の出力ベクトル（512次元）との**内積 + バイアス**です：

$$h^{{(2)}}_j = \\sum_{{i=0}}^{{511}} w^{{(2)}}_{{{0}i}} \\cdot \\text{{ReLU}}(h^{{(1)}}_i) + b^{{(2)}}_j$$

---

### ポイント：0の入力は計算に影響しない

ReLUで 0 になった入力は、どんな重みと掛けても **0** です。
今の入力では **{_n_active}/512個が活性（非ゼロ）**、**{512 - _n_active}個が不活性（ゼロ）** です。

| 入力 i | ReLU①出力 | 重み w[0][i] | 積（寄与） |
|:---:|:---:|:---:|:---:|
{_act_rows}
| ... | ... | ... | ... |
{_zero_rows}

---

### ニューロン0（j=0）の具体的な計算結果

$$h^{{(2)}}_0 = \\underbrace{{\\sum w_i \\cdot \\text{{ReLU}}(h^{{(1)}}_i)}}_{{= {_dot2_0:.4f}}} + \\underbrace{{b_0}}_{{= {_b2[0]:.4f}}} = {_h2_0:.4f}$$

これは `x_lin2[0]` の値（{float(x_lin2.squeeze()[0].item()):.4f}）と一致します。
""")

    mo.accordion({"📖 ReLU① → Linear② の計算詳細（クリックで展開）": _detail})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ3：Linear(512→512) → ReLU

    同じ構造でもう1層。前のReLU出力（正の値のみ）を入力として、再度512次元に変換します。
    2層重ねることで、より複雑な特徴の組み合わせを表現できます。
    """)
    return


@app.cell
def _(np, plt, x_lin2, x_relu2):
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 7),
                                gridspec_kw={"width_ratios": [2, 1]})

    lin2_np  = x_lin2.squeeze().numpy()
    relu2_np = x_relu2.squeeze().numpy()
    vmax_l2  = np.abs(lin2_np).max()

    ax_l2h = axes3[0, 0]
    im_l2 = ax_l2h.imshow(lin2_np.reshape(32, 16), cmap="RdBu_r",
                           vmin=-vmax_l2, vmax=vmax_l2, aspect="auto")
    fig3.colorbar(im_l2, ax=ax_l2h, shrink=0.8)
    ax_l2h.set_title("Linear(512→512) 後\nshape: (1, 512)", fontsize=10)
    ax_l2h.set_xlabel("列インデックス (0〜15)")
    ax_l2h.set_ylabel("行インデックス (0〜31)")

    ax_l2d = axes3[0, 1]
    ax_l2d.hist(lin2_np, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax_l2d.axvline(0, color="red", lw=1.5, linestyle="--", label="x=0")
    ax_l2d.set_title("値の分布", fontsize=10)
    ax_l2d.set_xlabel("活性化値")
    ax_l2d.set_ylabel("個数")
    ax_l2d.legend(fontsize=9)
    ax_l2d.grid(True, alpha=0.3)

    ax_r2h = axes3[1, 0]
    im_r2 = ax_r2h.imshow(relu2_np.reshape(32, 16), cmap="Oranges",
                           vmin=0, vmax=vmax_l2, aspect="auto")
    fig3.colorbar(im_r2, ax=ax_r2h, shrink=0.8)
    n_zero2 = int((relu2_np == 0).sum())
    ax_r2h.set_title(f"ReLU 後\nshape: (1, 512)  ← 不活性: {n_zero2}/512個", fontsize=10)
    ax_r2h.set_xlabel("列インデックス (0〜15)")
    ax_r2h.set_ylabel("行インデックス (0〜31)")

    ax_r2d = axes3[1, 1]
    pos2 = relu2_np[relu2_np > 0]
    ax_r2d.hist(pos2, bins=40, color="orangered", edgecolor="white", alpha=0.8, label="正（活性）")
    ax_r2d.bar(0, n_zero2, width=0.3, color="lightgray", edgecolor="gray", label=f"0（不活性: {n_zero2}個）")
    ax_r2d.set_title("値の分布（負がすべて0に）", fontsize=10)
    ax_r2d.set_xlabel("活性化値")
    ax_r2d.set_ylabel("個数")
    ax_r2d.legend(fontsize=9)
    ax_r2d.grid(True, alpha=0.3)

    plt.suptitle("STEP 3：Linear(512→512) + ReLU の変換", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig3
    return (
        ax_l2d, ax_l2h, ax_r2d, ax_r2h,
        axes3, fig3, im_l2, im_r2,
        lin2_np, n_zero2, pos2, relu2_np, vmax_l2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ3→4：最終層はなぜ ReLU がないのか？

    ```
    ReLU②出力 (1, 512)
      ↓
    Linear③：W₃(10×512) との内積 + b₃   ← 10クラス分のスコアを計算
      ↓
    Logits (1, 10)  ← ここにはReLUがない！
      ↓
    Softmax
    ```

    ### ReLUがない理由

    - 出力層の値（logits）はそのまま **Softmax に渡されます**
    - Softmax は $e^{z_i}$ を使うため、**負の値でも正しく処理できます**
    - むしろ ReLU で負のスコアを0にしてしまうと、「このクラスらしくない」という情報が失われてしまいます

    ### Softmax が確率に変換する仕組み

    $$P(\\text{class}=i) = \\frac{e^{z_i}}{\\sum_{j=0}^{9} e^{z_j}}$$

    | logit が大きい | → $e^{z_i}$ が大きい | → 確率が高い |
    |:---:|:---:|:---:|
    | logit が小さい（負） | → $e^{z_i}$ が小さい | → 確率が低い |

    負のlogitも $e^{z_i} > 0$ なので、確率として意味を持ちます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ステップ4：Linear(512→10) → Softmax → argmax = Predicted Class

    **出力層**：512次元 → 10クラスの生スコア（logits）に圧縮します。

    **Softmax**：生スコアを「合計が1になる確率」に変換します。

    $$P(\\text{class}=i) = \\frac{e^{z_i}}{\\sum_{j=0}^{9} e^{z_j}}$$

    **argmax**：確率が最も高いクラスのインデックスが **Predicted Class** です。
    """)
    return


@app.cell
def _(class_names, np, plt, x_logits, x_prob, y_pred):
    _class_names = class_names

    logits_np = x_logits.squeeze().numpy()
    prob_np   = x_prob.squeeze().numpy()

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 4))

    # --- Logits（生スコア）棒グラフ ---
    ax_log = axes4[0]
    _bar_colors = ["orangered" if i == y_pred else "steelblue" for i in range(10)]
    _bars = ax_log.bar(range(10), logits_np, color=_bar_colors, edgecolor="white")
    ax_log.set_xticks(range(10))
    ax_log.set_xticklabels(_class_names, rotation=30, ha="right", fontsize=9)
    ax_log.axhline(0, color="black", lw=0.8)
    ax_log.set_title("Linear(512→10) 後：Logits（生スコア）\nshape: (1, 10)　← まだ確率ではない", fontsize=10)
    ax_log.set_ylabel("スコア値")
    ax_log.grid(True, axis="y", alpha=0.3)
    for _i, (_b, _v) in enumerate(zip(_bars, logits_np)):
        ax_log.text(_b.get_x() + _b.get_width()/2,
                    _v + (0.05 if _v >= 0 else -0.1),
                    f"{_v:.2f}", ha="center", fontsize=7,
                    color="black")

    # --- Softmax確率 棒グラフ ---
    ax_prob = axes4[1]
    _bar_colors2 = ["orangered" if i == y_pred else "steelblue" for i in range(10)]
    _bars2 = ax_prob.bar(range(10), prob_np, color=_bar_colors2, edgecolor="white")
    ax_prob.set_xticks(range(10))
    ax_prob.set_xticklabels(_class_names, rotation=30, ha="right", fontsize=9)
    ax_prob.set_ylim(0, max(prob_np.max() * 1.25, 0.3))
    ax_prob.set_title(f"Softmax 後：クラス確率　（合計 = {prob_np.sum():.3f}）\n"
                      f"→ argmax = {y_pred}：「{_class_names[y_pred]}」が最も高い確率", fontsize=10)
    ax_prob.set_ylabel("確率")
    ax_prob.grid(True, axis="y", alpha=0.3)
    for _i, (_b, _v) in enumerate(zip(_bars2, prob_np)):
        ax_prob.text(_b.get_x() + _b.get_width()/2, _v + 0.003,
                     f"{_v:.3f}", ha="center", fontsize=7)

    # 最大値に星マーク
    ax_prob.annotate(f"★ Predicted\nClass: {y_pred}",
                     xy=(y_pred, prob_np[y_pred]),
                     xytext=(y_pred + (2 if y_pred < 7 else -2), prob_np[y_pred] + 0.05),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     fontsize=9, color="red", fontweight="bold")

    plt.suptitle(f"STEP 4：出力層 → Softmax → Predicted Class: {y_pred}（{_class_names[y_pred]}）",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig4
    return (
        _class_names, ax_log, ax_prob, axes4,
        fig4, logits_np, prob_np,
    )


@app.cell(hide_code=True)
def _(mo, np, x_logits, x_prob):
    _logits_np = x_logits.squeeze().numpy()
    _prob_np   = x_prob.squeeze().numpy()
    _logits_sum = float(_logits_np.sum())
    _prob_sum   = float(_prob_np.sum())

    _logit_rows = "\n".join(
        f"| {_i} | {_v:.4f} |"
        for _i, _v in enumerate(_logits_np)
    )

    _q1 = mo.md("""
### ✅ はい、正確な理解です

W[10, 512] の各行は「1クラス分の採点基準（512次元）」です。

```
W[0] (512次元) → 「0らしさ」を採点する重みベクトル
W[1] (512次元) → 「1らしさ」を採点する重みベクトル
...
W[9] (512次元) → 「9らしさ」を採点する重みベクトル
```

512個のニューロン出力（特徴ベクトル）と各クラスの重みベクトルの**内積**を取ることで、
「この特徴の組み合わせはクラス j にどれだけ近いか」のスコアが得られます。

| | 第1層 W[512, 784] | 出力層 W[10, 512] |
|:---|:---:|:---:|
| **1行の意味** | 1種類の画像フィルタ | 1クラスの採点基準 |
| **出力の意味** | 各フィルタへの反応度 | 各クラスのスコア（logit） |
""")

    _q2 = mo.md(f"""
### ✅ はい、合計は1になりません

Linear(512→10) の出力（logits）はただの「生スコア」なので、合計は1になる保証がありません。

| クラス | logit |
|:---:|:---:|
{_logit_rows}
| **合計** | **{_logits_sum:.4f}**（≠ 1） |

だからこそ **Softmax** が必要になります。

$$P(\\text{{class}}=j) = \\frac{{e^{{\\text{{logit}}_j}}}}{{\\sum_{{k=0}}^{{9}} e^{{\\text{{logit}}_k}}}}$$

Softmax を適用すると全クラスの確率の合計が必ず 1.000 になります（現在の入力では合計 = **{_prob_sum:.4f}**）。

| | logits（Linear出力） | 確率（Softmax後） |
|:---|:---:|:---:|
| **値の範囲** | $-\\infty$ 〜 $+\\infty$ | 0 〜 1 |
| **合計** | 1になる保証なし | **必ず 1** |
| **意味** | 相対的なスコア | 確率として解釈できる |

> argmax は logits でも確率でも**同じクラスを指します**（Softmax は大小関係を変えない）。
> Softmax が必要なのは「確率として解釈したい」「損失関数（Cross Entropy）の計算に使う」ためです。
""")

    mo.accordion({
        "❓ W[10,512] は「512の特徴からどのクラスに最も近いかを計算する行列」という理解で合っていますか？": _q1,
        "❓ 各クラスの計算結果（logits）は合計しても 1 にならない？": _q2,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 全ステップのデータ形状まとめ
    """)
    return


@app.cell
def _(np, plt, x_flat, x_lin1, x_lin2, x_logits, x_prob, x_relu1, x_relu2, X, y_pred):
    _steps = [
        ("入力 X",          X.shape,          X.squeeze().numpy().flatten()),
        ("Flatten後",       x_flat.shape,     x_flat.squeeze().numpy()),
        ("Linear1後",       x_lin1.shape,     x_lin1.squeeze().numpy()),
        ("ReLU1後",         x_relu1.shape,    x_relu1.squeeze().numpy()),
        ("Linear2後",       x_lin2.shape,     x_lin2.squeeze().numpy()),
        ("ReLU2後",         x_relu2.shape,    x_relu2.squeeze().numpy()),
        ("Logits",          x_logits.shape,   x_logits.squeeze().numpy()),
        ("Softmax確率",     x_prob.shape,     x_prob.squeeze().numpy()),
    ]

    fig5, ax5 = plt.subplots(figsize=(13, 5))
    ax5.axis("off")

    _col_labels = ["ステップ", "shape", "次元数", "最小値", "最大値", "平均値", "ゼロの個数"]
    _rows = []
    for _name, _shape, _vals in _steps:
        _n_zero = int((np.array(_vals) == 0).sum())
        _rows.append([
            _name,
            str(tuple(_shape)),
            str(int(np.prod(_shape))),
            f"{_vals.min():.4f}",
            f"{_vals.max():.4f}",
            f"{_vals.mean():.4f}",
            str(_n_zero),
        ])

    _table = ax5.table(
        cellText=_rows,
        colLabels=_col_labels,
        cellLoc="center",
        loc="center",
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(10)
    _table.scale(1, 1.8)

    # ヘッダー行を強調
    for _col in range(len(_col_labels)):
        _table[0, _col].set_facecolor("#2c5f8a")
        _table[0, _col].set_text_props(color="white", fontweight="bold")

    # ReLU行をオレンジに
    for _row_i in [4, 6]:
        for _col in range(len(_col_labels)):
            _table[_row_i, _col].set_facecolor("#fff3e0")

    ax5.set_title(f"各ステップのデータ形状と統計値　（Predicted Class: {y_pred}）",
                  fontsize=12, fontweight="bold", pad=15)
    fig5
    return _col_labels, _rows, _steps, _table, ax5, fig5


if __name__ == "__main__":
    app.run()
