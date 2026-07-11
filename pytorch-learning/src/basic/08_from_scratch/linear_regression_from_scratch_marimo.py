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
    # ゼロから作る線形回帰：生テンソル → autograd → nn.Module

    チュートリアルでは `nn.Module` や `optimizer` を「使う」ことを学びました。
    このノートブックでは、それらを **一切使わずに** 学習を実装し、
    段階的にPyTorchの部品へ置き換えていきます。

    | ステージ | 勾配計算 | パラメータ更新 | モデル定義 |
    |:---:|:---|:---|:---|
    | ① 生テンソル | **手計算した数式** | 自分で `w -= lr * grad` | 自分で `w*x + b` |
    | ② autograd | `loss.backward()` | 自分で `w -= lr * w.grad` | 自分で `w*x + b` |
    | ③ フルPyTorch | `loss.backward()` | `optimizer.step()` | `nn.Linear` |

    3つとも **まったく同じ計算** をしています。
    「PyTorchが裏で何をやっているか」を①と②で体感するのが目的です。

    ```
    [参考] What is torch.nn really?
    https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
    ```
    """)
    return (mo,)


# ============================================================
# インポート
# ============================================================
@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from torch import nn

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    return nn, np, plt, torch


# ============================================================
# Section 1: 問題設定
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. 問題設定：直線をデータに合わせる

    真の関係 $y = 2.0\,x - 1.0$ にノイズを加えたデータを作り、
    パラメータ $w, b$ を **知らないふりをして** データから推定します。

    モデルと損失関数（平均二乗誤差 MSE）：

    $$
    \hat{y}_i = w x_i + b, \qquad
    L(w, b) = \frac{1}{N} \sum_{i=0}^{N-1} (\hat{y}_i - y_i)^2
    $$
    """)
    return


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(0.0, 2.0, step=0.1, value=0.5, label="ノイズの大きさ σ")
    n_slider = mo.ui.slider(10, 200, step=10, value=50, label="データ点数 N")
    mo.hstack([noise_slider, n_slider], justify="start", gap=3)
    return n_slider, noise_slider


@app.cell
def _(n_slider, noise_slider, torch):
    TRUE_W, TRUE_B = 2.0, -1.0

    gen = torch.Generator().manual_seed(42)
    x_data = torch.rand(n_slider.value, generator=gen) * 4 - 2  # [-2, 2]
    y_data = TRUE_W * x_data + TRUE_B + noise_slider.value * torch.randn(
        n_slider.value, generator=gen
    )
    return TRUE_B, TRUE_W, x_data, y_data


@app.cell
def _(TRUE_B, TRUE_W, plt, x_data, y_data):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.scatter(x_data, y_data, alpha=0.6, label="観測データ")
    _xs = [-2, 2]
    _ax.plot(_xs, [TRUE_W * _x + TRUE_B for _x in _xs], "r--", lw=2,
             label=f"真の直線 y = {TRUE_W}x + {TRUE_B}")
    _ax.set_title("学習データ：この点群に直線をフィットさせる", fontsize=12)
    _ax.set_xlabel("入力 x")
    _ax.set_ylabel("出力 y")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig
    return


# ============================================================
# Section 2: ステージ① 生テンソル + 手計算の勾配
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. ステージ①：勾配を紙とペンで求めて実装する

    autograd を使わないので、勾配 $\dfrac{\partial L}{\partial w}, \dfrac{\partial L}{\partial b}$
    を自分で導出する必要があります。連鎖律（Chain Rule）で展開すると：

    $$
    \frac{\partial L}{\partial w}
    = \frac{1}{N}\sum_i \underbrace{2(\hat{y}_i - y_i)}_{\partial L / \partial \hat{y}_i}
      \cdot \underbrace{x_i}_{\partial \hat{y}_i / \partial w}
    \qquad
    \frac{\partial L}{\partial b}
    = \frac{1}{N}\sum_i 2(\hat{y}_i - y_i) \cdot 1
    $$

    > **「誤差 (ŷ−y) に、そのパラメータが出力へ与える影響 (x や 1) を掛けて平均する」**

    これがまさに `loss.backward()` が自動でやっている計算です。
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(0.001, 0.5, step=0.001, value=0.1, label="学習率 lr")
    epochs_slider = mo.ui.slider(10, 300, step=10, value=100, label="エポック数")
    mo.hstack([lr_slider, epochs_slider], justify="start", gap=3)
    return epochs_slider, lr_slider


@app.cell
def _(epochs_slider, lr_slider, torch, x_data, y_data):
    def train_manual(x, y, lr, epochs):
        """ステージ①：勾配を手計算の数式で求める学習ループ"""
        w = torch.tensor(0.0)  # 初期値
        b = torch.tensor(0.0)
        history = []  # (w, b, loss) を各エポックで記録

        for _ in range(epochs):
            # --- ① 予測（Forward）：モデルも自分で書く ---
            y_hat = w * x + b
            # --- ② 損失（MSE）---
            loss = ((y_hat - y) ** 2).mean()
            # --- ③ 勾配：導出した数式をそのままコードに ---
            grad_w = (2 * (y_hat - y) * x).mean()
            grad_b = (2 * (y_hat - y)).mean()
            # --- ④ 更新：勾配の逆方向へ一歩 ---
            w = w - lr * grad_w
            b = b - lr * grad_b
            history.append((w.item(), b.item(), loss.item()))

        return history

    hist_manual = train_manual(
        x_data, y_data, lr_slider.value, epochs_slider.value
    )
    return (hist_manual,)


@app.cell
def _(epochs_slider, mo):
    epoch_view = mo.ui.slider(
        1, epochs_slider.value, step=1, value=epochs_slider.value,
        label="表示するエポック（学習の途中経過を見る）",
    )
    epoch_view
    return (epoch_view,)


@app.cell
def _(TRUE_B, TRUE_W, epoch_view, hist_manual, plt, x_data, y_data):
    _k = epoch_view.value - 1
    _w, _b, _loss = hist_manual[_k]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左：現在のフィット
    _ax1.scatter(x_data, y_data, alpha=0.5, label="データ")
    _xs = [-2, 2]
    _ax1.plot(_xs, [TRUE_W * _x + TRUE_B for _x in _xs], "r--", lw=1.5, label="真の直線")
    _ax1.plot(_xs, [_w * _x + _b for _x in _xs], "g-", lw=2.5,
              label=f"学習中: y = {_w:.2f}x + {_b:.2f}")
    _ax1.set_title(f"エポック {epoch_view.value}: 損失 = {_loss:.4f}", fontsize=12)
    _ax1.set_xlabel("入力 x")
    _ax1.set_ylabel("出力 y")
    _ax1.legend(fontsize=9)
    _ax1.grid(alpha=0.3)

    # 右：損失曲線
    _losses = [h[2] for h in hist_manual]
    _ax2.plot(_losses, lw=2)
    _ax2.axvline(_k, color="g", ls=":", lw=2, label=f"現在: エポック {epoch_view.value}")
    _ax2.set_title("損失の推移（手計算勾配による学習）", fontsize=12)
    _ax2.set_xlabel("エポック")
    _ax2.set_ylabel("MSE損失")
    _ax2.set_yscale("log")
    _ax2.legend(fontsize=9)
    _ax2.grid(alpha=0.3)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **操作ガイド**：
    > - 「表示するエポック」を動かすと、直線が徐々にデータへ吸い付いていく様子が見えます
    > - 学習率を **0.4 以上** にすると損失が振動・発散します（一歩が大きすぎて谷を飛び越える）
    > - 学習率を **0.005 以下** にすると収束が極端に遅くなります（エポック数を増やして確認）
    """)
    return


# ============================================================
# Section 3: ステージ② autograd に置き換える
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. ステージ②：勾配計算だけ autograd に任せる

    ステージ①との差分は **勾配の数式を消して `loss.backward()` にした** ことだけです。

    新しく登場する3つの約束事：

    | コード | なぜ必要か |
    |:---|:---|
    | `requires_grad=True` | 「このテンソルの勾配を追跡して」という宣言 |
    | `with torch.no_grad():` | 更新操作自体を計算グラフに記録させない |
    | `w.grad.zero_()` | `backward()` は勾配を **加算** するので、毎回リセットが必要 |
    """)
    return


@app.cell
def _(epochs_slider, lr_slider, torch, x_data, y_data):
    def train_autograd(x, y, lr, epochs):
        """ステージ②：勾配計算をautogradに任せる学習ループ"""
        w = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        losses = []

        for _ in range(epochs):
            y_hat = w * x + b                     # ① 予測（同じ）
            loss = ((y_hat - y) ** 2).mean()      # ② 損失（同じ）
            loss.backward()                       # ③ 勾配：数式が消えた！

            with torch.no_grad():                 # ④ 更新は追跡の外で
                w -= lr * w.grad
                b -= lr * b.grad
            w.grad.zero_()                        # 勾配をリセット
            b.grad.zero_()

            losses.append(loss.item())

        return w.item(), b.item(), losses

    w_auto, b_auto, losses_auto = train_autograd(
        x_data, y_data, lr_slider.value, epochs_slider.value
    )
    return b_auto, losses_auto, w_auto


# ============================================================
# Section 4: ステージ③ nn.Module + optimizer
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. ステージ③：モデルと更新もPyTorchの部品に置き換える

    最後に、残っていた手書き部分をすべて標準部品に置き換えます。

    | ステージ②の手書き | ステージ③の部品 |
    |:---|:---|
    | `w * x + b` | `nn.Linear(1, 1)` |
    | `((y_hat - y)**2).mean()` | `nn.MSELoss()` |
    | `w -= lr * w.grad` | `optimizer.step()` |
    | `w.grad.zero_()` | `optimizer.zero_grad()` |

    これがチュートリアル06章で見た「いつもの学習ループ」そのものです。
    """)
    return


@app.cell
def _(epochs_slider, lr_slider, nn, torch, x_data, y_data):
    def train_full_pytorch(x, y, lr, epochs):
        """ステージ③：フルPyTorch版の学習ループ"""
        model = nn.Linear(1, 1)  # w と b を内包
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        x_col = x.unsqueeze(1)  # nn.Linear は (N, 特徴数) を期待する
        y_col = y.unsqueeze(1)
        losses = []

        for _ in range(epochs):
            y_hat = model(x_col)          # ① 予測
            loss = loss_fn(y_hat, y_col)  # ② 損失
            loss.backward()               # ③ 勾配
            optimizer.step()              # ④ 更新
            optimizer.zero_grad()         # リセット
            losses.append(loss.item())

        w = model.weight.item()
        b = model.bias.item()
        return w, b, losses

    w_full, b_full, losses_full = train_full_pytorch(
        x_data, y_data, lr_slider.value, epochs_slider.value
    )
    return b_full, losses_full, w_full


@app.cell
def _(
    TRUE_B,
    TRUE_W,
    b_auto,
    b_full,
    hist_manual,
    losses_auto,
    losses_full,
    plt,
    w_auto,
    w_full,
):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.plot([h[2] for h in hist_manual], lw=4, alpha=0.5, label="① 生テンソル（手計算勾配）")
    _ax.plot(losses_auto, lw=2.5, ls="--", label="② autograd")
    _ax.plot(losses_full, lw=1.5, ls=":", color="k", label="③ nn.Module + optimizer")
    _ax.set_title("3ステージの損失曲線はほぼ一致する（同じ計算だから）", fontsize=12)
    _ax.set_xlabel("エポック")
    _ax.set_ylabel("MSE損失")
    _ax.set_yscale("log")
    _ax.legend()
    _ax.grid(alpha=0.3)

    _w1, _b1, _ = hist_manual[-1]
    print(f"真の値          : w = {TRUE_W:.4f}, b = {TRUE_B:.4f}")
    print(f"① 手計算勾配    : w = {_w1:.4f}, b = {_b1:.4f}")
    print(f"② autograd     : w = {w_auto:.4f}, b = {b_auto:.4f}")
    print(f"③ フルPyTorch  : w = {w_full:.4f}, b = {b_full:.4f}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > ③が①②と微妙にずれることがあるのは、`nn.Linear` の初期値が
    > $w=b=0$ ではなくランダム初期化だからです（ずれ方も含めて観察してみてください）。
    """)
    return


# ============================================================
# まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## まとめ

    | 性質 | ① 生テンソル | ② autograd | ③ フルPyTorch |
    |:---:|:---:|:---:|:---:|
    | 勾配の求め方 | 手で導出した数式 | `loss.backward()` | `loss.backward()` |
    | 更新 | `w -= lr * grad` | `w -= lr * w.grad` | `optimizer.step()` |
    | モデル | 式を直接書く | 式を直接書く | `nn.Linear` |
    | 書く量 | 多い | 中 | 少ない |
    | 学べること | 勾配降下法の本質 | autogradの約束事 | 実務の標準形 |

    **次のステップ**：同じフォルダの `fashionmnist_scratch_marimo.py` で、
    今度は **多クラス分類（FashionMNIST）** をautogradだけで書いてみましょう。
    モデルが行列積になり、損失がクロスエントロピーになりますが、骨格は同じです。
    """)
    return


if __name__ == "__main__":
    app.run()
