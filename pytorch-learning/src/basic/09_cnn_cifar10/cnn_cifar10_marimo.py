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
    # CNNで画像分類：畳み込みの仕組みとCIFAR-10

    FashionMNISTの線形モデルには限界がありました（正解率85%前後で頭打ち）。
    理由は「画像を784次元ベクトルに **平らにした瞬間、ピクセルの位置関係が失われる**」からです。

    **畳み込みニューラルネットワーク（CNN）** は、
    小さなフィルタを画像上でスライドさせることで位置関係を保ったまま特徴を抽出します。

    このノートブックの流れ：

    1. 畳み込み演算そのものをインタラクティブに理解する
    2. MaxPoolingの役割を見る
    3. CIFAR-10（カラー写真10クラス）でCNNを実際に学習する
    """)
    return (mo,)


# ============================================================
# インポート
# ============================================================
@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets
    from torchvision.transforms import v2
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"使用デバイス: {device}")
    return DataLoader, F, Subset, datasets, device, nn, np, plt, torch, v2


# ============================================================
# Section 1: 畳み込み演算とは
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. 畳み込み演算：フィルタを滑らせて「パターン検出器」を作る

    ### 計算方法

    $3 \times 3$ のカーネル $K$ を画像 $X$ 上でスライドさせ、各位置で **要素積の総和** を取ります。

    $$
    Y[i, j] = \sum_{u=0}^{2} \sum_{v=0}^{2} X[i+u,\ j+v] \cdot K[u, v]
    $$

    ### 何を表現しているか？

    > **「カーネルと似たパターンがある場所ほど、出力が大きくなる（パターン検出器）」**

    - カーネルが縦エッジ検出型 → 縦の輪郭がある場所だけ光る
    - カーネルがぼかし型 → 出力は滑らかになる

    線形分類器では $W$ が「クラス全体のテンプレート」でしたが、
    CNNのカーネルは「**局所的な** パターンのテンプレート」です。
    しかも画像のどこにあっても同じカーネルで検出できます（位置不変性）。
    """)
    return


@app.cell
def _(datasets, mo, torch, v2):
    # 畳み込みデモ用にFashionMNIST画像を1枚使う
    _demo_ds = datasets.FashionMNIST(
        root="data", train=True, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    demo_img = _demo_ds[1][0][0]  # ズボンの画像 (28, 28)

    kernel_choice = mo.ui.dropdown(
        options=["縦エッジ検出", "横エッジ検出", "ぼかし（平均）", "シャープ化"],
        value="縦エッジ検出",
        label="カーネルの種類",
    )
    kernel_choice
    return demo_img, kernel_choice


@app.cell
def _(F, demo_img, kernel_choice, plt, torch):
    KERNELS = {
        "縦エッジ検出": torch.tensor([[-1., 0., 1.],
                                     [-2., 0., 2.],
                                     [-1., 0., 1.]]),
        "横エッジ検出": torch.tensor([[-1., -2., -1.],
                                     [0., 0., 0.],
                                     [1., 2., 1.]]),
        "ぼかし（平均）": torch.ones(3, 3) / 9,
        "シャープ化": torch.tensor([[0., -1., 0.],
                                   [-1., 5., -1.],
                                   [0., -1., 0.]]),
    }
    _kernel = KERNELS[kernel_choice.value]

    # conv2d は (バッチ, チャネル, 高さ, 幅) を期待する
    _out = F.conv2d(
        demo_img.view(1, 1, 28, 28), _kernel.view(1, 1, 3, 3), padding=1
    )[0, 0]

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(12, 4))

    _ax1.imshow(demo_img, cmap="gray")
    _ax1.set_title("入力画像（28×28）", fontsize=11)
    _ax1.axis("off")

    _ax2.imshow(_kernel, cmap="RdBu_r", vmin=-2, vmax=2)
    for _u in range(3):
        for _v in range(3):
            _ax2.text(_v, _u, f"{_kernel[_u, _v]:.2f}", ha="center", va="center",
                      fontsize=11)
    _ax2.set_title(f"カーネル：{kernel_choice.value}（3×3）", fontsize=11)
    _ax2.axis("off")

    _ax3.imshow(_out, cmap="gray")
    _ax3.set_title("出力（畳み込み結果）", fontsize=11)
    _ax3.axis("off")

    _fig.suptitle("カーネルを画像上でスライドさせ、各位置で要素積の総和を取る", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **操作ガイド**：カーネルの種類を切り替えて出力の変化を観察してください。
    > - **縦エッジ検出**：ズボンの左右の輪郭（縦線）だけが白く光る
    > - **横エッジ検出**：裾や腰の水平な輪郭が光る
    > - **ぼかし**：全体が滑らかになる
    >
    > CNNの学習とは、**これらのカーネルの数値を自動で獲得すること** です。
    > 上のカーネルは人間が設計しましたが、`nn.Conv2d` の中の数値は
    > 勾配降下法でタスクに最適な形に育っていきます。
    """)
    return


# ============================================================
# Section 2: MaxPooling
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. MaxPooling：情報を要約して「少しのズレ」に強くする

    $2 \times 2$ の窓の中の **最大値だけ** を残して画像を半分に縮小します。

    $$
    Y[i, j] = \max_{u, v \in \{0, 1\}} X[2i+u,\ 2j+v]
    $$

    > **「そのあたりにパターンが『あったか』だけ覚えて、正確な位置は忘れる」**

    - 位置が1〜2ピクセルずれても出力がほぼ変わらない → 頑健性
    - 画像サイズが半分になる → 計算量削減、より広い範囲を見られる
    """)
    return


@app.cell
def _(F, demo_img, plt, torch):
    _pooled1 = F.max_pool2d(demo_img.view(1, 1, 28, 28), 2)
    _pooled2 = F.max_pool2d(_pooled1, 2)

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 4))
    for _ax, _im, _title in zip(
        _axes,
        [demo_img, _pooled1[0, 0], _pooled2[0, 0]],
        ["元画像（28×28）", "MaxPool 1回（14×14）", "MaxPool 2回（7×7）"],
    ):
        _ax.imshow(_im, cmap="gray")
        _ax.set_title(_title, fontsize=11)
        _ax.axis("off")
    _fig.suptitle("解像度は下がるが「ズボンらしさ」は7×7でもまだ分かる", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# ============================================================
# Section 3: CIFAR-10 データセット
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. CIFAR-10：カラー写真の10クラス分類

    - 32×32ピクセルの **カラー画像**（チャネルが3つ: RGB）
    - 学習5万枚・テスト1万枚
    - FashionMNISTより格段に難しい：背景があり、向き・色・大きさがバラバラ

    入力の形が `(N, 3, 32, 32)` になる点に注意してください（グレースケールは `(N, 1, 28, 28)` でした）。
    """)
    return


@app.cell
def _(datasets, torch, v2):
    cifar_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # CIFAR-10の各チャネルの平均・標準偏差で正規化（学習が安定する定番処理）
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    cifar_train = datasets.CIFAR10(
        root="data", train=True, download=True, transform=cifar_transform
    )
    cifar_test = datasets.CIFAR10(
        root="data", train=False, download=True, transform=cifar_transform
    )

    CIFAR_CLASSES = [
        "飛行機", "自動車", "鳥", "猫", "鹿",
        "犬", "カエル", "馬", "船", "トラック",
    ]
    print(f"学習データ: {len(cifar_train)}枚, テストデータ: {len(cifar_test)}枚")
    return CIFAR_CLASSES, cifar_test, cifar_train


@app.cell
def _(CIFAR_CLASSES, cifar_train, datasets, plt):
    # 正規化前の生画像を表示用に取得
    _raw = datasets.CIFAR10(root="data", train=True, download=False)

    _fig, _axes = plt.subplots(2, 8, figsize=(13, 3.8))
    for _i, _ax in enumerate(_axes.flat):
        _img, _label = _raw[_i]
        _ax.imshow(_img)
        _ax.set_title(CIFAR_CLASSES[_label], fontsize=9)
        _ax.axis("off")
    _fig.suptitle("CIFAR-10のサンプル：32×32と小さいが実写なので難しい", fontsize=12)
    _fig.tight_layout()
    _fig
    return


# ============================================================
# Section 4: CNNモデルの定義
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. CNNモデル：「畳み込み → 活性化 → プーリング」を積み重ねる

    定番の構成は **[Conv → BatchNorm → ReLU] × 数回 → Pool** の繰り返しです。

    ```
    入力 (3, 32, 32)
      │ Conv2d(3→32) + BN + ReLU     ← 局所パターン検出（エッジ・色）
      │ Conv2d(32→32) + BN + ReLU
      │ MaxPool                       → (32, 16, 16)
      │ Conv2d(32→64) + BN + ReLU    ← パターンの組み合わせ（模様・部品）
      │ Conv2d(64→64) + BN + ReLU
      │ MaxPool                       → (64, 8, 8)
      │ Flatten                       → 4096次元
      │ Linear(4096→10)              ← 最後だけ全結合で分類
    出力 logits (10,)
    ```

    | 部品 | 役割 |
    |:---:|:---|
    | `Conv2d` | 学習可能なカーネルでパターン検出。チャネル数=カーネルの種類数 |
    | `BatchNorm2d` | 各チャネルの出力を平均0・分散1に整えて学習を安定・高速化 |
    | `ReLU` | 非線形性を入れる（これがないと何層積んでも線形モデルと同じ） |
    | `MaxPool2d` | 縮小して位置ズレに強く、視野を広く |
    """)
    return


@app.cell
def _(nn):
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 32x32 → 16x16
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 16x16 → 8x8
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    _n_params = sum(p.numel() for p in SimpleCNN().parameters())
    print(f"パラメータ数: {_n_params:,}")
    return (SimpleCNN,)


# ============================================================
# Section 5: 学習
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. 学習：ループの書き方はいつもと同じ

    学習ループは06章と同一です。CNNだからといって特別なことはありません。

    > ⏱️ フルデータ（5万枚）の学習は1エポック数分かかるため、
    > まず **サブセット（1万枚）× 3エポック** で流れを確認するのがおすすめです。
    > 時間があるときにフルデータへ切り替えてみてください（正解率が大きく上がります）。
    """)
    return


@app.cell
def _(mo):
    cnn_subset = mo.ui.dropdown(
        options={"サブセット（1万枚・お試し）": 10000, "フルデータ（5万枚）": 50000},
        value="サブセット（1万枚・お試し）",
        label="学習データ量",
    )
    cnn_epochs = mo.ui.slider(1, 10, step=1, value=3, label="エポック数")
    cnn_train_btn = mo.ui.run_button(label="CNNを学習する")
    mo.hstack([cnn_subset, cnn_epochs, cnn_train_btn], justify="start", gap=2)
    return cnn_epochs, cnn_subset, cnn_train_btn


@app.cell
def _(
    DataLoader,
    SimpleCNN,
    Subset,
    cifar_test,
    cifar_train,
    cnn_epochs,
    cnn_subset,
    cnn_train_btn,
    device,
    mo,
    nn,
    torch,
):
    mo.stop(
        not cnn_train_btn.value,
        mo.md("> ⏸️ 「CNNを学習する」ボタンを押すと学習が始まります"),
    )

    def evaluate_model(model, loader):
        """テストデータでの正解率を計算する"""
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch).argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += len(y_batch)
        return correct / total

    def train_cnn(n_train, epochs):
        torch.manual_seed(0)
        train_subset = Subset(cifar_train, range(n_train))
        train_dl = DataLoader(train_subset, batch_size=128, shuffle=True)
        test_dl = DataLoader(cifar_test, batch_size=512)

        model = SimpleCNN().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses, accs = [], []
        for _epoch in mo.status.progress_bar(
            range(epochs), title="学習中", subtitle="エポック"
        ):
            model.train()
            for x_batch, y_batch in train_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
            accs.append(evaluate_model(model, test_dl))

        return model, losses, accs

    cnn_model, cnn_losses, cnn_accs = train_cnn(
        cnn_subset.value, cnn_epochs.value
    )
    print(f"最終テスト正解率: {cnn_accs[-1]:.1%}")
    return cnn_accs, cnn_losses, cnn_model


@app.cell
def _(cnn_accs, cnn_losses, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    _ax1.plot(cnn_losses, alpha=0.35, lw=0.8)
    _window = 30
    if len(cnn_losses) > _window:
        _smooth = [
            sum(cnn_losses[_i:_i + _window]) / _window
            for _i in range(len(cnn_losses) - _window)
        ]
        _ax1.plot(range(_window // 2, _window // 2 + len(_smooth)), _smooth, lw=2)
    _ax1.set_title("学習損失の推移", fontsize=11)
    _ax1.set_xlabel("バッチ番号")
    _ax1.set_ylabel("クロスエントロピー損失")
    _ax1.grid(alpha=0.3)

    _ax2.plot(range(1, len(cnn_accs) + 1), [_a * 100 for _a in cnn_accs],
              "o-", lw=2, markersize=8)
    _ax2.axhline(10, color="gray", ls="--", label="当てずっぽう（10%）")
    _ax2.set_title("テスト正解率の推移", fontsize=11)
    _ax2.set_xlabel("エポック")
    _ax2.set_ylabel("正解率（%）")
    _ax2.set_xticks(range(1, len(cnn_accs) + 1))
    _ax2.legend(fontsize=9)
    _ax2.grid(alpha=0.3)

    _fig.tight_layout()
    _fig
    return


# ============================================================
# Section 6: 予測の中身を見る
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. 予測を確認する

    テスト画像に対する予測を見てみます。間違えたものは
    「猫⇔犬」「鹿⇔馬」のような、人間でも紛らわしいペアが多いはずです。
    """)
    return


@app.cell
def _(CIFAR_CLASSES, cnn_model, datasets, device, mo, plt, torch, v2):
    pred_btn = mo.ui.run_button(label="別のサンプルを見る")

    _raw_test = datasets.CIFAR10(root="data", train=False, download=False)
    _norm = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    _idx = torch.randperm(len(_raw_test))[:8]
    _imgs = [_raw_test[_i][0] for _i in _idx]
    _labels = [_raw_test[_i][1] for _i in _idx]
    _batch = torch.stack([_norm(_im) for _im in _imgs]).to(device)

    cnn_model.eval()
    with torch.no_grad():
        _preds = cnn_model(_batch).argmax(dim=1).cpu()

    _fig, _axes = plt.subplots(1, 8, figsize=(14, 2.6))
    for _i, _ax in enumerate(_axes):
        _ax.imshow(_imgs[_i])
        _ok = _preds[_i].item() == _labels[_i]
        _ax.set_title(
            f"予測: {CIFAR_CLASSES[_preds[_i]]}\n正解: {CIFAR_CLASSES[_labels[_i]]}",
            fontsize=8, color="green" if _ok else "red",
        )
        _ax.axis("off")
    _fig.suptitle("CNNの予測（緑: 正解, 赤: 不正解）", fontsize=11)
    _fig.tight_layout()
    mo.vstack([pred_btn, _fig])
    return


# ============================================================
# Section 7: 学習された第1層カーネルの可視化
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. 学習されたカーネルを覗く

    第1層の32個のカーネル（3×3×RGB）を可視化します。
    Section 1 で人間が設計した「縦エッジ検出」のようなパターンを、
    ネットワークが **勾配降下法だけで自力獲得している** ことが分かります。
    """)
    return


@app.cell
def _(cnn_model, plt):
    _kernels = cnn_model.features[0].weight.detach().cpu()  # (32, 3, 3, 3)
    # RGBの3チャネルを0-1に正規化して色付きで表示
    _kmin, _kmax = _kernels.min(), _kernels.max()
    _kernels_norm = (_kernels - _kmin) / (_kmax - _kmin)

    _fig, _axes = plt.subplots(4, 8, figsize=(11, 5.5))
    for _j, _ax in enumerate(_axes.flat):
        _ax.imshow(_kernels_norm[_j].permute(1, 2, 0))  # (3,3,3) → (3,3,RGB)
        _ax.axis("off")
    _fig.suptitle("第1層の学習済みカーネル32個：エッジ検出器や色検出器が自然に生まれる",
                  fontsize=12)
    _fig.tight_layout()
    _fig
    return


# ============================================================
# まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## まとめ

    | 性質 | 線形モデル（08章） | CNN（本章） |
    |:---:|:---:|:---:|
    | 画像の扱い | 平らにする（位置情報を破壊） | 2次元のまま処理 |
    | パラメータ | クラスごとの全体テンプレート | 局所パターンのカーネル |
    | 位置ズレ | 弱い | 強い（カーネル共有＋Pooling） |
    | PyTorch | `nn.Linear` | `nn.Conv2d` + `nn.MaxPool2d` |
    | CIFAR-10正解率の目安 | 〜40% | 70%〜（フルデータ・10エポック） |

    **発展課題**：
    1. フルデータ・10エポックで学習し、正解率の変化を確認する
    2. `v2.RandomHorizontalFlip()` や `v2.RandomCrop(32, padding=4)` をtransformに追加（データ拡張）
    3. 畳み込みブロックをもう1段追加（64→128チャネル、4×4まで縮小）するとどうなるか

    **次のステップ**：`transfer_learning_marimo.py` で、
    ImageNetで事前学習済みのResNetを **転移学習** させ、
    「ゼロから学習」との違いを体感しましょう。
    """)
    return


if __name__ == "__main__":
    app.run()
