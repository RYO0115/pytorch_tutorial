import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


# ============================================================
# タイトル
# ============================================================
@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # パラメータの最適化（Optimizing Model Parameters）

    ニューラルネットワークの **「学習」** とは何でしょう？

    一言で言えば、**「正解に近い予測ができるようにパラメータ（重みとバイアス）を繰り返し調整すること」** です。

    このノートブックでは PyTorch の最適化チュートリアルを通して、
    学習がどういう手順で進み、各ステップで **何が・なぜ** 行われているかを直感的に理解します。

    ```
    [チュートリアル] https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    ```
    """)
    return (mo,)


# ============================================================
# インポート
# ============================================================
@app.cell
def _():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    return DataLoader, FancyBboxPatch, datasets, device, nn, np, plt, torch, v2


# ============================================================
# Section 1: 学習の全体像
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1. 学習の全体像：何を何度も繰り返しているのか？

    ニューラルネットワークの学習は、以下の **4ステップ** をデータ全体が尽きるまで繰り返します。

    | ステップ | 処理 | たとえで言うと |
    |:---:|:---|:---|
    | ① | 予測（Forward Pass） | 「これはスニーカーだ」と答える |
    | ② | 損失計算（Loss） | 「正解はサンダルだった。どれだけ外れたか」を数値化 |
    | ③ | 逆伝播（Backward Pass） | 「どのパラメータをどう変えると次は正解に近づくか」を計算 |
    | ④ | パラメータ更新（Update） | 実際にパラメータを少しだけ修正 |

    これを全データで1周すると **1エポック** です。
    """)
    return


@app.cell
def _(FancyBboxPatch, plt):
    fig_loop, ax_loop = plt.subplots(figsize=(13, 4.5))
    ax_loop.set_xlim(0, 13)
    ax_loop.set_ylim(0, 4.5)
    ax_loop.axis("off")

    _steps = [
        (1.8, 2.5, "① 予測\n(Forward Pass)", "#d0e8ff",
         "model(X)\nlogits を出力"),
        (4.8, 2.5, "② 損失計算\n(Loss Function)", "#ffe0b2",
         "loss_fn(pred, y)\n外れ具合をスカラーで"),
        (7.8, 2.5, "③ 逆伝播\n(Backward Pass)", "#ffd5d5",
         "loss.backward()\n各パラメータの勾配を計算"),
        (10.8, 2.5, "④ パラメータ更新\n(Optimizer Step)", "#c8f7c5",
         "optimizer.step()\nw ← w - lr × grad"),
    ]
    for _cx, _cy, _label, _color, _sub in _steps:
        _box = FancyBboxPatch((_cx - 1.3, _cy - 0.65), 2.6, 1.3,
                              boxstyle="round,pad=0.1", facecolor=_color,
                              edgecolor="#888", lw=1.5)
        ax_loop.add_patch(_box)
        ax_loop.text(_cx, _cy + 0.1, _label, ha="center", va="center",
                    fontsize=10, fontweight="bold")
        ax_loop.text(_cx, _cy - 1.15, _sub, ha="center", va="center",
                    fontsize=8, color="#555")

    for _i in range(len(_steps) - 1):
        _x1 = _steps[_i][0] + 1.3
        _x2 = _steps[_i + 1][0] - 1.3
        ax_loop.annotate("", xy=(_x2, 2.5), xytext=(_x1, 2.5),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # ④ の下辺から下へ → 水平に左へ → ① の下辺に矢印で戻るU字パス
    _y_path = 0.55          # ボックス下辺(1.85)より十分下
    _y_box_bottom = 1.85    # 2.5 - 0.65
    ax_loop.plot([10.8, 10.8], [_y_box_bottom, _y_path], color="purple", lw=2)
    ax_loop.plot([10.8, 1.8],  [_y_path, _y_path],       color="purple", lw=2)
    ax_loop.annotate("", xy=(1.8, _y_box_bottom), xytext=(1.8, _y_path),
                     arrowprops=dict(arrowstyle="->", lw=2, color="purple"))
    ax_loop.text(6.3, 0.28,
                "全バッチを処理したら 1エポック完了。これをエポック数分繰り返す",
                ha="center", va="center", fontsize=9, color="purple",
                bbox=dict(facecolor="white", edgecolor="purple",
                          boxstyle="round,pad=0.25", alpha=0.85))

    plt.title("学習ループの全体像：4ステップを全バッチ × 全エポック分繰り返す",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_loop
    return (ax_loop, fig_loop)


# ============================================================
# Section 2: データとモデルの準備
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. データとモデルの準備

    今回は **FashionMNIST**（衣類画像の 10 クラス分類）を使います。
    28×28 ピクセルのグレースケール画像を、T-shirt / Sneaker / Bag などに分類するタスクです。

    ```python
    training_data = datasets.FashionMNIST(root="data", train=True,  ...)
    test_data     = datasets.FashionMNIST(root="data", train=False, ...)
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader  = DataLoader(test_data,     batch_size=64)
    ```

    **DataLoader** は大きなデータセットを `batch_size` 枚ずつに分けて読み込む道具です。
    訓練データ 60,000 枚 ÷ 64 = 約 **938 バッチ** になります。
    """)
    return


@app.cell
def _(DataLoader, datasets, device, nn, torch, v2):
    _transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=_transform
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=_transform
    )
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader  = DataLoader(test_data,     batch_size=64)

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
            return self.linear_relu_stack(x)

    _ = device  # suppress lint
    return NeuralNetwork, test_data, test_dataloader, train_dataloader, training_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### データセットの中身を確認する

    FashionMNIST は衣類の画像データセットです。
    **28×28 ピクセルのグレースケール画像** が 10 カテゴリに分類されています。

    | データ種別 | 枚数 | 用途 |
    |:---:|:---:|:---|
    | 訓練データ | 60,000 枚 | パラメータの学習に使う |
    | テストデータ | 10,000 枚 | 学習後の性能評価に使う（学習には使わない） |

    訓練データとテストデータで **同じカテゴリ** の画像がどんなものか、実際に見てみましょう。
    スライダーでサンプルを切り替えられます。
    """)
    return


@app.cell
def _(mo):
    sample_seed_slider = mo.ui.slider(
        start=0, stop=500, step=1, value=0,
        label="サンプル番号（0〜500）  ← 動かすと別の画像を表示",
    )
    sample_seed_slider
    return (sample_seed_slider,)


@app.cell
def _(plt, sample_seed_slider, test_data, torch, training_data):
    _class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    _n_classes = len(_class_names)
    _offset = sample_seed_slider.value

    # 各クラスの代表サンプルを訓練・テストそれぞれから1枚ずつ取得
    _train_samples = {}
    _test_samples  = {}

    _train_idx = 0
    for _img, _lbl in training_data:
        if _lbl not in _train_samples:
            _train_samples[_lbl] = (_img, _train_idx)
        _train_idx += 1
        if len(_train_samples) == _n_classes:
            break

    _test_idx = 0
    for _img, _lbl in test_data:
        if _lbl not in _test_samples:
            _test_samples[_lbl] = (_img, _test_idx)
        _test_idx += 1
        if len(_test_samples) == _n_classes:
            break

    # スライダー分ずらしたランダムサンプル（訓練データ）
    torch.manual_seed(_offset)
    _rand_indices = torch.randperm(len(training_data))[:_n_classes * 3]

    fig_data, _axes = plt.subplots(3, _n_classes, figsize=(15, 5.5))
    fig_data.patch.set_facecolor("#f8f8f8")

    _row_labels = ["訓練データ\n（各クラス代表）", "テストデータ\n（各クラス代表）", "訓練データ\n（ランダム）"]

    for _col in range(_n_classes):
        # 行0: 訓練データ 各クラス代表
        _ax0 = _axes[0, _col]
        _tr_img, _tr_idx = _train_samples[_col]
        _ax0.imshow(_tr_img.squeeze(), cmap="gray", interpolation="nearest")
        _ax0.set_title(f"{_class_names[_col]}", fontsize=7.5, pad=3)
        _ax0.axis("off")

        # 行1: テストデータ 各クラス代表
        _ax1 = _axes[1, _col]
        _te_img, _te_idx = _test_samples[_col]
        _ax1.imshow(_te_img.squeeze(), cmap="gray", interpolation="nearest")
        _ax1.set_title(f"idx={_te_idx}", fontsize=6.5, color="#666", pad=3)
        _ax1.axis("off")

        # 行2: 訓練データ ランダムサンプル
        _ax2 = _axes[2, _col]
        _ri = _rand_indices[_col].item()
        _r_img, _r_lbl = training_data[_ri]
        _ax2.imshow(_r_img.squeeze(), cmap="gray", interpolation="nearest")
        _ax2.set_title(f"{_class_names[_r_lbl]}", fontsize=7.5,
                      color="#c0392b" if _r_lbl != _col else "#27ae60", pad=3)
        _ax2.axis("off")

    # 行ラベルを左端に追加
    for _r, _rl in enumerate(_row_labels):
        _axes[_r, 0].set_ylabel(_rl, fontsize=8.5, labelpad=4,
                                rotation=0, ha="right", va="center")

    # 境界線
    for _col in range(_n_classes):
        for _r, _color in [(0, "#d0e8ff"), (1, "#ffe0b2"), (2, "#f0f0f0")]:
            _axes[_r, _col].set_facecolor(_color)
            for _sp in _axes[_r, _col].spines.values():
                _sp.set_visible(True)
                _sp.set_edgecolor("#ccc")
                _sp.set_linewidth(0.8)

    plt.suptitle(
        "FashionMNIST データセットのサンプル画像（28×28ピクセル、グレースケール）\n"
        "青行=訓練データ代表  /  橙行=テストデータ代表  /  グレー行=ランダム（スライダーで切替）",
        fontsize=10, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig_data
    return (fig_data,)


# ============================================================
# Section 3: ハイパーパラメータ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. ハイパーパラメータ：「学習のルール」を決める設定値

    **ハイパーパラメータ** は、学習が始まる前に人間が決める設定値です。
    モデルが自動で学ぶ「重み・バイアス」とは違い、**学習の方法そのものを決める数値**です。

    ```python
    learning_rate = 1e-3   # 学習率
    batch_size    = 64     # バッチサイズ
    epochs        = 10     # エポック数
    ```

    | ハイパーパラメータ | 意味 | 例 |
    |:---|:---|:---:|
    | **学習率 (lr)** | 1 ステップでパラメータをどれだけ動かすか | `1e-3` |
    | **バッチサイズ** | 一度に処理するサンプル数 | `64` |
    | **エポック数** | 全データを何周学習するか | `10` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 学習率が大きい・小さいと何が起きる？（クリックで展開）": mo.md(r"""
### 学習率（Learning Rate）の直感

パラメータ更新の式：

$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$$

$\alpha$（アルファ）が学習率です。

| 学習率 $\alpha$ | パラメータの動き | 問題 |
|:---:|:---|:---|
| **大きすぎる** | 一度に大きく動く | 最適値を飛び越えて損失が発散することも |
| **小さすぎる** | 少しずつしか動かない | 収束が非常に遅くなる |
| **適切** | ちょうど良いペースで進む | バランスよく学習 |

### バッチサイズとエポック数の直感

- **バッチサイズ 64**：60,000 枚 ÷ 64 = 約 938 バッチで 1 エポック
- **エポック数 10**：同じデータで 10 周練習する

1 エポックで 938 回パラメータが更新されます。
10 エポックでは **合計 9,380 回** の更新が行われます。
""")
    })
    return


# ============================================================
# Section 4: 損失関数
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. 損失関数：「どれだけ外れているか」を数値化する

    **損失関数（Loss Function）** はモデルの予測が正解からどれだけズレているかを
    1 つのスカラー値（損失）で表します。**損失が小さいほど予測が正解に近い**ことを意味します。

    分類問題には **`nn.CrossEntropyLoss`** を使います：

    ```python
    loss_fn = nn.CrossEntropyLoss()
    ```

    内部では次の 2 つを組み合わせています：
    1. **Softmax**：モデルの raw 出力（logits）を「各クラスの確率」に変換
    2. **負の対数尤度（NLL）**：正解クラスの確率が高いほど損失が小さくなる計算
    """)
    return


@app.cell
def _(np, plt, torch):
    fig_loss, axes_loss = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左: -log(p) の曲線
    _ax_l = axes_loss[0]
    _probs = np.linspace(0.01, 0.99, 200)
    _ax_l.plot(_probs, -np.log(_probs), color="orangered", lw=2.5)
    _ax_l.fill_between(_probs, -np.log(_probs), 0, alpha=0.1, color="orangered")
    for _p, _col, _lbl in [
        (0.05, "#c0392b", "確率5%→損失 大\n（ほぼ間違い）"),
        (0.5,  "#e67e22", "確率50%→損失 中\n（五分五分）"),
        (0.95, "#27ae60", "確率95%→損失 小\n（ほぼ正解）"),
    ]:
        _ax_l.scatter([_p], [-np.log(_p)], color=_col, s=80, zorder=5)
        _ax_l.annotate(
            _lbl, xy=(_p, -np.log(_p)),
            xytext=(_p + 0.07, -np.log(_p) + (0.5 if _p < 0.5 else -0.8)),
            fontsize=8.5, color=_col,
            arrowprops=dict(arrowstyle="->", color=_col, lw=1.2),
        )
    _ax_l.set_xlabel("正解クラスの予測確率（Softmax後）")
    _ax_l.set_ylabel("損失 = −log(確率)")
    _ax_l.set_title("CrossEntropyLoss の直感\n「正解クラスの確率が高いほど損失は小さい」", fontsize=10)
    _ax_l.set_xlim(0, 1)
    _ax_l.grid(True, alpha=0.3)

    # 右: 10クラスの出力例
    _ax_r = axes_loss[1]
    _class_names = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    torch.manual_seed(7)
    _logits = torch.randn(10)
    _softmax = torch.softmax(_logits, dim=0).numpy()
    _true_cls = 7  # Sneaker

    _colors_bar = ["#4CAF50" if i == _true_cls else "#90CAF9" for i in range(10)]
    _bars = _ax_r.bar(range(10), _softmax, color=_colors_bar, edgecolor="white")
    _ax_r.set_xticks(range(10))
    _ax_r.set_xticklabels(_class_names, rotation=45, ha="right", fontsize=8)
    _ax_r.set_ylabel("Softmax 確率")
    _loss_val = float(-np.log(_softmax[_true_cls]))
    _ax_r.set_title(
        f"モデルの出力例（正解クラス: {_class_names[_true_cls]}）\n"
        f"正解クラスの確率 = {_softmax[_true_cls]:.3f} → 損失 = {_loss_val:.3f}",
        fontsize=10,
    )
    _ax_r.axhline(0.1, color="gray", linestyle="--", alpha=0.6, label="ランダム確率(1/10)")
    _ax_r.legend(fontsize=8)
    _ax_r.text(
        _true_cls, _softmax[_true_cls] + 0.005,
        f"正解\n{_softmax[_true_cls]:.3f}",
        ha="center", fontsize=8, color="#2e7d32", fontweight="bold",
    )
    _ax_r.grid(True, axis="y", alpha=0.3)

    plt.suptitle("損失関数（CrossEntropyLoss）のしくみ", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_loss
    return (axes_loss, fig_loss)


# ============================================================
# Section 5: オプティマイザの3ステップ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. オプティマイザの 3 ステップ：パラメータを更新するエンジン

    **オプティマイザ** は「勾配（gradient）を使ってパラメータを更新する役割」を担います。

    今回は **SGD（確率的勾配降下法）** を使います：

    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ```

    1 バッチごとに次の **3 ステップ** を実行します：

    ```python
    loss.backward()        # ① 勾配を計算（逆伝播）
    optimizer.step()       # ② パラメータを更新
    optimizer.zero_grad()  # ③ 勾配をリセット（次のバッチのために）
    ```
    """)
    return


@app.cell
def _(FancyBboxPatch, plt):
    fig_opt, ax_opt = plt.subplots(figsize=(13, 5.5))
    ax_opt.set_xlim(0, 13)
    ax_opt.set_ylim(0, 5.5)
    ax_opt.axis("off")

    _opt_items = [
        (2.2, 3.2, "① loss.backward()\n勾配を計算", "#ffd5d5",
         "各パラメータ w に対して\n∂loss/∂w を自動計算し\n.grad 属性に書き込む"),
        (6.5, 3.2, "② optimizer.step()\nパラメータを更新", "#ffe0b2",
         "w  ←  w − lr × w.grad\n勾配の逆方向に\n少しだけパラメータを動かす"),
        (10.8, 3.2, "③ optimizer.zero_grad()\n勾配をリセット", "#c8f7c5",
         "w.grad を 0 にクリア\n次のバッチの計算に\n前回の勾配を混ぜないために必須"),
    ]
    for _cx, _cy, _label, _color, _desc in _opt_items:
        _box = FancyBboxPatch((_cx - 1.8, _cy - 0.75), 3.6, 1.5,
                              boxstyle="round,pad=0.1", facecolor=_color,
                              edgecolor="#888", lw=1.5)
        ax_opt.add_patch(_box)
        ax_opt.text(_cx, _cy + 0.15, _label, ha="center", va="center",
                   fontsize=10, fontweight="bold")
        ax_opt.text(_cx, _cy - 1.5, _desc, ha="center", va="center",
                   fontsize=8.5, color="#444")

    for _i in range(len(_opt_items) - 1):
        _x1 = _opt_items[_i][0] + 1.8
        _x2 = _opt_items[_i + 1][0] - 1.8
        ax_opt.annotate("", xy=(_x2, 3.2), xytext=(_x1, 3.2),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # why zero_grad 注釈
    ax_opt.text(6.5, 5.1,
               "なぜ zero_grad() が必要か？",
               ha="center", fontsize=10, color="#c0392b", fontweight="bold")
    ax_opt.text(6.5, 4.75,
               "PyTorch は backward() のたびに勾配を「累積（加算）」する仕様。\n"
               "リセットしないと前のバッチの勾配が混ざり、誤った方向に更新される。",
               ha="center", fontsize=8.5, color="#555")

    plt.title("1 バッチあたりの 3 ステップ：勾配計算 → 更新 → リセット",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_opt
    return (ax_opt, fig_opt)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 zero_grad() を忘れると何が起きるか？（クリックで展開）": mo.md("""
### 勾配累積の問題

```python
# ❌ zero_grad を忘れた場合
for X, y in dataloader:
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()       # 勾配が累積される！
    optimizer.step()      # 古い勾配も含んだ誤った更新に

# ✅ 正しい書き方
for X, y in dataloader:
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()        # ① 勾配計算
    optimizer.step()       # ② パラメータ更新
    optimizer.zero_grad()  # ③ 次のバッチのために勾配クリア
```

### なぜ「累積」がデフォルトなのか？

意図的に複数バッチの勾配を合算してから更新する
**「勾配累積（Gradient Accumulation）」** というテクニックがあるためです。
GPU メモリが少ない環境で大きなバッチを模擬できます。
""")
    })
    return


# ============================================================
# Section 6: train_loop の詳細
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6. トレーニングループ（train_loop）のコード解説

    ```python
    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)   # 訓練データ総数（60,000）
        model.train()                    # ← 訓練モードに切り替え ★重要★

        for batch, (X, y) in enumerate(dataloader):  # バッチごとに処理
            pred = model(X)              # ① 予測（Forward Pass）
            loss = loss_fn(pred, y)      # ② 損失計算

            loss.backward()              # ③ 逆伝播（勾配計算）
            optimizer.step()             # ④ パラメータ更新
            optimizer.zero_grad()        # ⑤ 勾配リセット（次バッチのために）

            if batch % 100 == 0:
                print(f"loss: {loss.item():>7f}  [{batch * 64:>5d}/{size:>5d}]")
    ```

    ### `model.train()` が必要な理由

    | メソッド | 使う場面 | 影響するレイヤー |
    |:---:|:---:|:---|
    | `model.train()` | 学習時 | **Dropout**（ランダム無効化）・**BatchNorm**（統計更新）が有効 |
    | `model.eval()` | 評価・推論時 | Dropout 無効（全ニューロン使用）・BatchNorm 固定 |

    今回のモデルには Dropout がありませんが、**習慣として必ず書く**ことが重要です。
    """)
    return


# ============================================================
# Section 7: test_loop の詳細
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 7. テストループ（test_loop）のコード解説

    ```python
    def test_loop(dataloader, model, loss_fn):
        model.eval()                    # ← 評価モードに切り替え ★重要★
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():           # ← 勾配計算を無効化 ★重要★
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Accuracy: {100*correct:.1f}%,  Avg loss: {test_loss:.4f}")
    ```

    ### `torch.no_grad()` が必要な理由

    テスト時はパラメータを更新しないので **勾配の計算は不要** です。
    `torch.no_grad()` で計算グラフの記録を止めることで、**メモリを節約し処理を高速化**できます。

    ### なぜ「訓練していないデータ」で評価するか？

    訓練データだけで正確率を測ると、**過学習（Overfitting）** を見逃してしまいます。
    → 訓練データは完璧に解けるが、未知のデータでは全く解けない状態
    テストデータで評価することで「本当の汎化能力」を測ります。
    """)
    return


# ============================================================
# Section 8: インタラクティブ学習
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 8. 実際に学習させて、値の変化を見てみよう

    学習率とエポック数を変えると、損失・正答率がどう変わるか確認しましょう。

    > **注意**：設定を変えると FashionMNIST の学習が再実行されます（CPU で数分かかることがあります）。
    """)
    return


@app.cell
def _(mo):
    lr_selector = mo.ui.dropdown(
        options=["0.0001", "0.001", "0.005", "0.01", "0.05"],
        value="0.001",
        label="学習率 (lr)  ← 大きいと速いが不安定、小さいと安定だが遅い",
    )
    epoch_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=2,
        label="エポック数  ← 多いほど学習が進む（時間もかかる）",
    )
    mo.vstack([lr_selector, epoch_slider])
    return epoch_slider, lr_selector


@app.cell
def _(
    NeuralNetwork,
    device,
    epoch_slider,
    lr_selector,
    nn,
    test_dataloader,
    torch,
    train_dataloader,
):
    def _run_train_loop(dataloader, model, loss_fn, optimizer):
        model.train()
        ep_loss, ep_correct, ep_batches = 0.0, 0, 0
        batch_loss_log = []
        for _X, _y in dataloader:
            _X, _y = _X.to(device), _y.to(device)
            _pred = model(_X)
            _loss = loss_fn(_pred, _y)
            _loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ep_loss    += _loss.item()
            ep_correct += (_pred.argmax(1) == _y).float().sum().item()
            ep_batches += 1
            batch_loss_log.append(_loss.item())
        return (
            ep_loss / ep_batches,
            ep_correct / len(dataloader.dataset) * 100,
            batch_loss_log,
        )

    def _run_test_loop(dataloader, model, loss_fn):
        model.eval()
        te_loss, te_correct = 0.0, 0
        with torch.no_grad():
            for _X, _y in dataloader:
                _X, _y = _X.to(device), _y.to(device)
                _pred = model(_X)
                te_loss    += loss_fn(_pred, _y).item()
                te_correct += (_pred.argmax(1) == _y).float().sum().item()
        return (
            te_loss / len(dataloader),
            te_correct / len(dataloader.dataset) * 100,
        )

    _lr     = float(lr_selector.value)
    _epochs = epoch_slider.value
    _model  = NeuralNetwork().to(device)
    _loss_fn   = nn.CrossEntropyLoss()
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_lr)

    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []
    first_epoch_batch_losses = []

    for _ep in range(_epochs):
        _tl, _ta, _bl = _run_train_loop(
            train_dataloader, _model, _loss_fn, _optimizer
        )
        _vl, _va = _run_test_loop(test_dataloader, _model, _loss_fn)
        train_losses.append(_tl)
        train_accs.append(_ta)
        test_losses.append(_vl)
        test_accs.append(_va)
        if _ep == 0:
            first_epoch_batch_losses = _bl

    return (
        first_epoch_batch_losses,
        test_accs,
        test_losses,
        train_accs,
        train_losses,
    )


@app.cell
def _(
    epoch_slider,
    first_epoch_batch_losses,
    lr_selector,
    np,
    plt,
    test_accs,
    test_losses,
    train_accs,
    train_losses,
):
    _lr_val = float(lr_selector.value)
    _epochs = epoch_slider.value
    _ep_range = list(range(1, _epochs + 1))

    fig_res, _axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── 左: 1エポック目のバッチごとの損失 ──────────────────────
    _ax1 = _axes[0]
    _raw = first_epoch_batch_losses
    _w   = 30
    _smooth = np.convolve(_raw, np.ones(_w) / _w, mode="valid")
    _ax1.plot(range(len(_raw)), _raw,
              color="steelblue", lw=0.7, alpha=0.35, label="各バッチの損失")
    _ax1.plot(range(len(_smooth)), _smooth,
              color="steelblue", lw=2.2, label=f"移動平均（{_w}バッチ）")
    _ax1.set_xlabel("バッチ番号（1エポック目）")
    _ax1.set_ylabel("損失")
    _ax1.set_title(
        "1エポック目：バッチごとの損失の変化\n"
        "「細かく揺れながらも全体として下がる」のが正常",
        fontsize=10,
    )
    _ax1.legend(fontsize=8)
    _ax1.grid(True, alpha=0.3)

    # ── 中: エポックごとの損失 ──────────────────────────────
    _ax2 = _axes[1]
    _ax2.plot(_ep_range, train_losses, "o-", color="steelblue",
              lw=2, ms=6, label="訓練損失")
    _ax2.plot(_ep_range, test_losses,  "s--", color="orangered",
              lw=2, ms=6, label="テスト損失")
    for _i in range(_epochs):
        _ax2.annotate(
            f"{train_losses[_i]:.3f}", xy=(_i + 1, train_losses[_i]),
            xytext=(0, 7), textcoords="offset points",
            fontsize=8, color="steelblue", ha="center",
        )
        _ax2.annotate(
            f"{test_losses[_i]:.3f}", xy=(_i + 1, test_losses[_i]),
            xytext=(0, -14), textcoords="offset points",
            fontsize=8, color="orangered", ha="center",
        )
    _ax2.set_xlabel("エポック")
    _ax2.set_ylabel("平均損失")
    _ax2.set_title(
        f"エポックごとの損失推移  (lr={_lr_val})\n"
        "訓練・テスト損失ともに下がれば順調",
        fontsize=10,
    )
    _ax2.legend(fontsize=8)
    _ax2.set_xticks(_ep_range)
    _ax2.grid(True, alpha=0.3)

    # ── 右: エポックごとの正答率 ──────────────────────────────
    _ax3 = _axes[2]
    _ax3.plot(_ep_range, train_accs, "o-", color="steelblue",
              lw=2, ms=6, label="訓練正答率")
    _ax3.plot(_ep_range, test_accs,  "s--", color="orangered",
              lw=2, ms=6, label="テスト正答率")
    _ax3.axhline(10, color="gray", linestyle=":", lw=1.2,
                label="ランダム予測（10%）")
    for _i in range(_epochs):
        _ax3.annotate(
            f"{train_accs[_i]:.1f}%", xy=(_i + 1, train_accs[_i]),
            xytext=(0, 7), textcoords="offset points",
            fontsize=8, color="steelblue", ha="center",
        )
        _ax3.annotate(
            f"{test_accs[_i]:.1f}%", xy=(_i + 1, test_accs[_i]),
            xytext=(0, -14), textcoords="offset points",
            fontsize=8, color="orangered", ha="center",
        )
    _ax3.set_xlabel("エポック")
    _ax3.set_ylabel("正答率（%）")
    _ax3.set_title(
        f"エポックごとの正答率推移  (lr={_lr_val})\n"
        f"最終テスト正答率：{test_accs[-1]:.1f}%",
        fontsize=10,
    )
    _ax3.legend(fontsize=8)
    _ax3.set_xticks(_ep_range)
    _ax3.set_ylim(0, 100)
    _ax3.grid(True, alpha=0.3)

    plt.suptitle(
        f"学習結果サマリー（lr={_lr_val},  epochs={_epochs}）",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_res
    return (fig_res,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### グラフの読み方

    | グラフ | 何を示しているか | 「良い状態」の見え方 |
    |:---:|:---|:---|
    | **左**（バッチ損失） | 1エポック内の細かい変化 | 揺れながらも全体として右下がり |
    | **中**（エポック損失） | エポックを重ねた進捗 | 訓練・テスト損失が両方減少 |
    | **右**（正答率） | モデルの分類能力 | 10%（ランダム）より大きく上昇 |

    #### 試してみよう

    | 操作 | 観察できること |
    |:---|:---|
    | lr を `0.05` に上げる | 損失が不安定になったり発散することがある |
    | lr を `0.0001` に下げる | 損失の下がり方が非常に遅くなる |
    | エポック数を増やす | 損失が下がり続け、正答率が上昇する |
    """)
    return


# ============================================================
# Section 9: 1バッチの詳細ウォークスルー
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 9. 1 バッチの中で何が起きているか（具体的な数値で確認）

    実際に 1 バッチを取り出して、**パラメータ更新の前後で値がどう変わるか**を確認します。
    """)
    return


@app.cell
def _(NeuralNetwork, device, mo, nn, torch, train_dataloader):
    torch.manual_seed(0)
    _wk_model    = NeuralNetwork().to(device)
    _wk_loss_fn  = nn.CrossEntropyLoss()
    _wk_optimizer = torch.optim.SGD(_wk_model.parameters(), lr=1e-3)

    _X_wk, _y_wk = next(iter(train_dataloader))
    _X_wk, _y_wk = _X_wk.to(device), _y_wk.to(device)

    # ── 更新前の状態
    _wk_model.train()
    _pred_before = _wk_model(_X_wk)
    _loss_before = _wk_loss_fn(_pred_before, _y_wk)
    _acc_before  = (_pred_before.argmax(1) == _y_wk).float().mean().item() * 100
    _w1_before   = list(_wk_model.parameters())[0].data.clone()

    # ── backward → step
    _loss_before.backward()
    _first_layer = list(_wk_model.parameters())[0]
    _grad_norm   = _first_layer.grad.norm().item()
    _wk_optimizer.step()

    # ── 更新後の状態
    _w1_after   = list(_wk_model.parameters())[0].data.clone()
    _pred_after = _wk_model(_X_wk)
    _loss_after = _wk_loss_fn(_pred_after, _y_wk)
    _acc_after  = (_pred_after.argmax(1) == _y_wk).float().mean().item() * 100

    _delta_max  = (_w1_after - _w1_before).abs().max().item()
    _delta_mean = (_w1_after - _w1_before).abs().mean().item()

    mo.md(f"""
    ### 1 バッチ（64 サンプル）処理の前後

    | タイミング | 損失 | バッチ正答率 |
    |:---:|:---:|:---:|
    | **更新前**（ランダム初期値） | `{_loss_before.item():.4f}` | `{_acc_before:.1f}%` |
    | **更新後**（1 ステップ後） | `{_loss_after.item():.4f}` | `{_acc_after:.1f}%` |
    | **変化量** | `{_loss_before.item() - _loss_after.item():.4f}` 減少 | `{_acc_after - _acc_before:+.1f}%` |

    ### 第 1 層の重み（shape: 512 × 784）の変化

    | 指標 | 値 | 意味 |
    |:---|:---:|:---|
    | 勾配ノルム（‖∂L/∂W₁‖） | `{_grad_norm:.4f}` | この層がどれだけ損失に影響しているか |
    | 重みの最大変化量 | `{_delta_max:.6f}` | lr({0.001}) × 勾配の大きさ |
    | 重みの平均変化量 | `{_delta_mean:.6f}` | 全重みの平均的な更新幅 |

    ---

    > **ポイント**：1 バッチでの損失変化はごくわずかです。
    > これを **938 バッチ × エポック数** 分繰り返すことで、
    > 徐々に損失が下がり正答率が上がっていきます。
    > 「塵も積もれば山となる」のが勾配降下法の本質です。
    """)
    return


@app.cell
def _(NeuralNetwork, device, nn, np, plt, torch, train_dataloader):
    torch.manual_seed(0)
    _viz_model = NeuralNetwork().to(device)
    _viz_loss_fn = nn.CrossEntropyLoss()
    _viz_opt = torch.optim.SGD(_viz_model.parameters(), lr=1e-3)

    # 最初の 200 バッチでパラメータの変化を追跡
    _n_steps = 200
    _step_losses   = []
    _step_w1_norms = []
    _viz_model.train()
    for _b_idx, (_Xv, _yv) in enumerate(train_dataloader):
        if _b_idx >= _n_steps:
            break
        _Xv, _yv = _Xv.to(device), _yv.to(device)
        _pv = _viz_model(_Xv)
        _lv = _viz_loss_fn(_pv, _yv)
        _lv.backward()
        _viz_opt.step()
        _viz_opt.zero_grad()
        _step_losses.append(_lv.item())
        _step_w1_norms.append(list(_viz_model.parameters())[0].data.norm().item())

    fig_wk, axes_wk = plt.subplots(1, 2, figsize=(13, 4))

    _ax_a = axes_wk[0]
    _sw = 10
    _sl = np.convolve(_step_losses, np.ones(_sw) / _sw, mode="valid")
    _ax_a.plot(range(_n_steps), _step_losses,
               color="orangered", lw=0.8, alpha=0.35, label="各バッチ損失")
    _ax_a.plot(range(len(_sl)), _sl,
               color="orangered", lw=2.2, label=f"移動平均（{_sw}バッチ）")
    _ax_a.set_xlabel("バッチ番号（先頭200バッチ）")
    _ax_a.set_ylabel("損失")
    _ax_a.set_title("最初の 200 バッチでの損失変化\n更新を重ねるほど損失が低下していく", fontsize=10)
    _ax_a.legend(fontsize=8)
    _ax_a.grid(True, alpha=0.3)

    _ax_b = axes_wk[1]
    _ax_b.plot(range(_n_steps), _step_w1_norms,
               color="steelblue", lw=2, label="第1層の重みノルム ‖W₁‖")
    _ax_b.set_xlabel("バッチ番号（先頭200バッチ）")
    _ax_b.set_ylabel("重みノルム")
    _ax_b.set_title("第 1 層の重みノルムの変化\nパラメータが更新されるにつれて徐々に変化", fontsize=10)
    _ax_b.legend(fontsize=8)
    _ax_b.grid(True, alpha=0.3)

    plt.suptitle("1 バッチごとの更新が積み重なって学習が進む", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_wk
    return (axes_wk, fig_wk)


# ============================================================
# Section 10: まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## まとめ

    | 概念 | コード | 意味 |
    |:---|:---|:---|
    | ハイパーパラメータ | `lr=1e-3, batch_size=64, epochs=10` | 学習のルールを人間が設定する値 |
    | 損失関数 | `nn.CrossEntropyLoss()` | 予測のズレを 1 つの数値で表す（小さいほど良い） |
    | オプティマイザ | `torch.optim.SGD(model.parameters(), lr)` | 勾配を使ってパラメータを更新するエンジン |
    | 逆伝播 | `loss.backward()` | 各パラメータの「影響度（勾配）」を自動計算 |
    | パラメータ更新 | `optimizer.step()` | `w ← w − lr × grad`、勾配の逆方向に動かす |
    | 勾配リセット | `optimizer.zero_grad()` | 前のバッチの勾配が残らないように毎回必須 |
    | 訓練モード | `model.train()` | Dropout・BatchNorm を学習用設定に |
    | 評価モード | `model.eval()` | Dropout 無効・BatchNorm 固定 |
    | 勾配無効化 | `torch.no_grad()` | テスト時に勾配計算を省略（高速・省メモリ） |

    ### 学習ループの完全な形

    ```python
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):                        # ← エポック数分繰り返す
        # ── 訓練 ──────────────────────────────────────
        model.train()
        for X, y in train_dataloader:                 # ← バッチごとに処理
            pred = model(X)                           # ① 予測
            loss = loss_fn(pred, y)                   # ② 損失計算
            loss.backward()                           # ③ 勾配計算（逆伝播）
            optimizer.step()                          # ④ パラメータ更新
            optimizer.zero_grad()                     # ⑤ 勾配リセット

        # ── 評価 ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            for X, y in test_dataloader:
                pred      = model(X)
                test_loss = loss_fn(pred, y)
                correct   = (pred.argmax(1) == y).float().sum()
    ```

    このパターンを理解すれば、どんなニューラルネットワークの学習コードも読めるようになります！
    """)
    return


if __name__ == "__main__":
    app.run()
