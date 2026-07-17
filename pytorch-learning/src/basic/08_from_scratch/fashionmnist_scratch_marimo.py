import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


# ============================================================
# タイトル
# ============================================================
@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md(r"""
    # ゼロから作るFashionMNIST分類：nn.Module を使わない多クラス分類

    前のノートブック（線形回帰）の考え方を、そのまま **10クラス分類** に拡張します。
    使ってよいのは `torch.Tensor` と autograd（`backward()`）だけ。
    `nn.Module`・`nn.CrossEntropyLoss`・`optimizer` は使いません。

    変わるのは3点だけです：

    | | 線形回帰 | FashionMNIST分類 |
    |:---:|:---|:---|
    | モデル | $\hat{y} = wx + b$（スカラー） | $Z = XW + b$（**行列積**） |
    | 出力 | 実数1個 | 10クラス分のスコア（logits） |
    | 損失 | MSE | **クロスエントロピー**（自作する） |

    学習ループの骨格（予測 → 損失 → backward → 更新）は完全に同じです。
    """)
    return (mo,)


# ============================================================
# インポートとデータ
# ============================================================
@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    return DataLoader, datasets, plt, torch, v2


@app.cell
def _(DataLoader, datasets, torch, v2):
    _transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_ds = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=_transform
    )
    test_ds = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=_transform
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    CLASS_NAMES = [
        "Tシャツ", "ズボン", "セーター", "ドレス", "コート",
        "サンダル", "シャツ", "スニーカー", "バッグ", "ブーツ",
    ]
    return CLASS_NAMES, test_loader, train_loader


# ============================================================
# Section 1: モデルを行列積で書く
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. モデル：たった1つの行列積

    画像（28×28）を784次元ベクトルに平らにして、重み行列 $W$ を掛けます。

    $$
    \underbrace{Z}_{(N,\,10)} = \underbrace{X}_{(N,\,784)} \underbrace{W}_{(784,\,10)} + \underbrace{b}_{(10,)}
    $$

    - $W$ の **第 $j$ 列** は「クラス $j$ らしさを測るテンプレート」
    - $Z[i,j]$ は「画像 $i$ とテンプレート $j$ の内積」＝ クラス $j$ のスコア（logit）

    ## 2. 損失：クロスエントロピーを自作する

    スコアを確率に変換（softmax）し、正解クラスの確率の対数にマイナスを付けます。

    $$
    p_{ij} = \frac{e^{Z_{ij}}}{\sum_k e^{Z_{ik}}}, \qquad
    L = -\frac{1}{N}\sum_i \log p_{i,\,y_i}
    $$

    > **「正解クラスに割り当てた確率が低いほど、大きな罰を与える」**

    数値安定性のため、実装では log-sum-exp の形にします（下のコード参照）。
    """)
    return


@app.cell
def _(torch):
    def model_fn(x_batch, W, b):
        """モデル：784次元に平らにして行列積するだけ"""
        x_flat = x_batch.view(x_batch.shape[0], -1)  # (N, 1, 28, 28) → (N, 784)
        return x_flat @ W + b                        # (N, 10) の logits

    def cross_entropy_fn(logits, targets):
        """クロスエントロピーの自作実装（log-sum-exp で数値安定化）

        L_i = -Z[i, y_i] + log(Σ_k exp(Z[i, k]))
        """
        # そのまま exp すると簡単にオーバーフローするので最大値を引く
        z_max = logits.max(dim=1, keepdim=True).values
        log_sum_exp = z_max.squeeze(1) + (logits - z_max).exp().sum(dim=1).log()
        correct_score = logits[torch.arange(len(targets)), targets]
        return (log_sum_exp - correct_score).mean()

    # 検算：PyTorch組み込みと一致するか確認
    _logits = torch.randn(5, 10)
    _y = torch.randint(0, 10, (5,))
    _mine = cross_entropy_fn(_logits, _y)
    _official = torch.nn.functional.cross_entropy(_logits, _y)
    print(f"自作クロスエントロピー : {_mine.item():.6f}")
    print(f"F.cross_entropy       : {_official.item():.6f}")
    print(f"一致: {torch.allclose(_mine, _official)}")
    return cross_entropy_fn, model_fn


# ============================================================
# Section 2: 学習
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. 学習ループ：骨格は線形回帰と完全に同じ

    ミニバッチごとに「① 予測 → ② 損失 → ③ backward → ④ 手で更新」を回します。
    下のボタンで学習を開始してください（1エポック ≈ 数秒〜十数秒）。
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(0.01, 1.0, step=0.01, value=0.5, label="学習率 lr")
    epochs_slider = mo.ui.slider(1, 5, step=1, value=2, label="エポック数")
    train_btn = mo.ui.run_button(label="学習を実行")
    mo.hstack([lr_slider, epochs_slider, train_btn], justify="start", gap=2)
    return epochs_slider, lr_slider, train_btn


@app.cell
def _(
    cross_entropy_fn,
    epochs_slider,
    lr_slider,
    mo,
    model_fn,
    test_loader,
    torch,
    train_btn,
    train_loader,
):
    mo.stop(not train_btn.value, mo.md("> ⏸️ 「学習を実行」ボタンを押すと学習が始まります"))

    def evaluate(W, b, loader):
        """テストデータでの正解率を計算する"""
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                pred = model_fn(x_batch, W, b).argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += len(y_batch)
        return correct / total

    def train_scratch(lr, epochs):
        torch.manual_seed(0)
        # パラメータは自分で作る（小さな乱数で初期化）
        W = (torch.randn(784, 10) * 0.01).requires_grad_()
        b = torch.zeros(10, requires_grad=True)

        batch_losses = []
        epoch_accs = []
        for _epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                logits = model_fn(x_batch, W, b)              # ① 予測
                loss = cross_entropy_fn(logits, y_batch)      # ② 損失（自作）
                loss.backward()                               # ③ 勾配

                with torch.no_grad():                         # ④ 手で更新
                    W -= lr * W.grad
                    b -= lr * b.grad
                W.grad.zero_()
                b.grad.zero_()
                batch_losses.append(loss.item())

            epoch_accs.append(evaluate(W, b, test_loader))

        return W, b, batch_losses, epoch_accs

    W_trained, b_trained, batch_losses, epoch_accs = train_scratch(
        lr_slider.value, epochs_slider.value
    )
    print(f"最終テスト正解率: {epoch_accs[-1]:.1%}")
    return W_trained, b_trained, batch_losses, epoch_accs


@app.cell
def _(batch_losses, epoch_accs, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    _ax1.plot(batch_losses, alpha=0.4, lw=0.8)
    # 移動平均で滑らかに
    _window = 50
    if len(batch_losses) > _window:
        _smooth = [
            sum(batch_losses[_i:_i + _window]) / _window
            for _i in range(len(batch_losses) - _window)
        ]
        _ax1.plot(range(_window // 2, _window // 2 + len(_smooth)), _smooth, lw=2)
    _ax1.set_title("バッチごとの損失（薄い線）と移動平均（濃い線）", fontsize=11)
    _ax1.set_xlabel("バッチ番号")
    _ax1.set_ylabel("クロスエントロピー損失")
    _ax1.grid(alpha=0.3)

    _ax2.plot(range(1, len(epoch_accs) + 1), [_a * 100 for _a in epoch_accs],
              "o-", lw=2, markersize=8)
    _ax2.set_title("エポックごとのテスト正解率", fontsize=11)
    _ax2.set_xlabel("エポック")
    _ax2.set_ylabel("正解率（%）")
    _ax2.set_xticks(range(1, len(epoch_accs) + 1))
    _ax2.grid(alpha=0.3)

    _fig.tight_layout()
    _fig
    return


# ============================================================
# Section 3: 学習した重みの可視化
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. 学習した重み行列を「見る」

    $W$ の各列（784次元）を 28×28 に戻して画像として表示すると、
    **各クラスのテンプレート** が浮かび上がります。
    赤い部分＝「そこにピクセルがあるとこのクラスのスコアが上がる」領域です。
    """)
    return


@app.cell
def _(CLASS_NAMES, W_trained, plt):
    _fig, _axes = plt.subplots(2, 5, figsize=(12, 5.5))
    for _j, _ax in enumerate(_axes.flat):
        _template = W_trained[:, _j].detach().view(28, 28)
        _v = _template.abs().max()
        _ax.imshow(_template, cmap="RdBu_r", vmin=-_v, vmax=_v)
        _ax.set_title(CLASS_NAMES[_j], fontsize=10)
        _ax.axis("off")
    _fig.suptitle("重み行列 W の各列 ＝ 各クラスの「テンプレート」（赤: 正の寄与, 青: 負の寄与）",
                  fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **観察ポイント**：ズボンの列には縦長の2本線、ブーツには靴の輪郭が
    > うっすら見えるはずです。線形モデルは「画像とテンプレートの内積」しか
    > できないので、これが表現力の限界です（正解率も85%前後で頭打ち）。
    """)
    return


# ============================================================
# Section 4: 予測を見てみる
# ============================================================
@app.cell
def _(CLASS_NAMES, W_trained, b_trained, mo, model_fn, plt, test_loader, torch):
    sample_btn = mo.ui.run_button(label="別のサンプルを見る")

    _x, _y = next(iter(test_loader))
    _idx = torch.randperm(len(_x))[:8]
    with torch.no_grad():
        _logits = model_fn(_x[_idx], W_trained, b_trained)
        _preds = _logits.argmax(dim=1)

    _fig, _axes = plt.subplots(1, 8, figsize=(14, 2.4))
    for _i, _ax in enumerate(_axes):
        _ax.imshow(_x[_idx[_i], 0], cmap="gray")
        _ok = _preds[_i] == _y[_idx[_i]]
        _ax.set_title(
            f"予測: {CLASS_NAMES[_preds[_i]]}\n正解: {CLASS_NAMES[_y[_idx[_i]]]}",
            fontsize=8, color="green" if _ok else "red",
        )
        _ax.axis("off")
    _fig.suptitle("テストデータでの予測（緑: 正解, 赤: 不正解）", fontsize=11)
    _fig.tight_layout()
    mo.vstack([sample_btn, _fig])
    return


# ============================================================
# まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## まとめ

    | 部品 | 今回の自作版 | PyTorch標準版 |
    |:---:|:---|:---|
    | モデル | `x @ W + b` | `nn.Linear(784, 10)` |
    | 損失 | log-sum-exp を自分で実装 | `nn.CrossEntropyLoss()` |
    | 更新 | `W -= lr * W.grad` | `optim.SGD(...).step()` |
    | 正解率 | 約84〜85% | 同じ（同じ計算だから） |

    **確認クイズ**（自力で書けたら合格）：
    1. 何も見ずにこの学習ループを再現できますか？
    2. 隠れ層を1枚足す（`x @ W1 + b1` → ReLU → `@ W2 + b2`）と正解率はどう変わる？
    3. `W.grad.zero_()` を消すと何が起きる？（実際に消して観察してみてください）

    **次のステップ**：`09_cnn_cifar10/` で、線形モデルの限界を超える
    **畳み込みニューラルネットワーク（CNN）** に進みましょう。
    """)
    return


if __name__ == "__main__":
    app.run()
