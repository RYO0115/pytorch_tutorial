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
    # 転移学習：事前学習済みResNetを自分のタスクに流用する

    前のノートブックではCNNを **ゼロから** 学習しました。
    しかし実務では、ImageNet（120万枚・1000クラス）で学習済みのモデルを
    **出発点として流用する** のが標準です。これが **転移学習（Transfer Learning）** です。

    > **「エッジ・模様・物体部品の検出器はどんな画像タスクでも共通。
    > 学習済みのものを借りて、最後の分類部分だけ自分のタスクに合わせる」**

    ## 2つの流儀

    | 方式 | 凍結する範囲 | 学習が速い | 精度 | 使いどころ |
    |:---:|:---|:---:|:---:|:---|
    | **特徴抽出** | 最終層以外すべて凍結 | ◎ | ○ | データが少ない・タスクがImageNetに近い |
    | **ファインチューニング** | 凍結なし（全体を小さいlrで再学習） | △ | ◎ | データがそこそこある |

    ```
    [参考] https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"使用デバイス: {device}")
    return DataLoader, Subset, datasets, device, models, nn, plt, torch, v2


# ============================================================
# Section 1: 事前学習済みモデルを覗く
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. 学習済みResNet-18を読み込む

    `torchvision.models` から **重み込みで** モデルを取得できます
    （初回はダウンロードが走ります。約45MB）。

    ResNet-18の構造は大きく2つに分かれます：

    ```
    ① 特徴抽出部（conv1 〜 layer4 〜 avgpool）
         画像 → 512次元の特徴ベクトル。「画像の意味の要約」を作る部分
    ② 分類ヘッド（fc）
         nn.Linear(512, 1000) ← ImageNetの1000クラス用
    ```

    転移学習では **②を自分のクラス数の新しい層に差し替える** だけです。
    """)
    return


@app.cell
def _(models, nn):
    def build_model(num_classes, freeze_backbone):
        """事前学習済みResNet-18の分類ヘッドを差し替える

        freeze_backbone=True  → 特徴抽出（fc以外の勾配を止める）
        freeze_backbone=False → ファインチューニング（全体を学習）
        """
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # fc を差し替える（新しい層は requires_grad=True で作られる）
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # 構造を確認：fc が 512 → 1000 になっている
    _pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print("差し替え前の分類ヘッド:", _pretrained.fc)
    _replaced = build_model(num_classes=10, freeze_backbone=True)
    print("差し替え後の分類ヘッド:", _replaced.fc)

    _trainable = sum(p.numel() for p in _replaced.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in _replaced.parameters())
    print(f"学習対象パラメータ: {_trainable:,} / 全体 {_total:,} "
          f"({_trainable / _total:.2%}) ← 特徴抽出モードでは最終層だけ")
    return (build_model,)


# ============================================================
# Section 2: データ準備
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. データ準備：事前学習時と同じ前処理に合わせる

    重要な約束事：**入力の前処理は事前学習時と揃える** 必要があります。

    - ImageNetの学習画像サイズ（224×224）にリサイズ
    - ImageNetの平均・標準偏差で正規化

    CIFAR-10は32×32なので7倍に拡大することになります（ぼやけますが機能します）。

    > ⏱️ 224×224への拡大で計算が重くなるため、このノートブックでは
    > **学習2,000枚・テスト2,000枚のサブセット** を使います。
    > 「データが少ないときこそ転移学習が効く」という状況の再現でもあります。
    """)
    return


@app.cell
def _(DataLoader, Subset, datasets, torch, v2):
    # ImageNet事前学習時と同じ前処理
    imagenet_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _train_full = datasets.CIFAR10(
        root="data", train=True, download=True, transform=imagenet_transform
    )
    _test_full = datasets.CIFAR10(
        root="data", train=False, download=True, transform=imagenet_transform
    )

    N_TRAIN, N_TEST = 2000, 2000
    tl_train_dl = DataLoader(
        Subset(_train_full, range(N_TRAIN)), batch_size=64, shuffle=True
    )
    tl_test_dl = DataLoader(Subset(_test_full, range(N_TEST)), batch_size=128)

    TL_CLASSES = [
        "飛行機", "自動車", "鳥", "猫", "鹿",
        "犬", "カエル", "馬", "船", "トラック",
    ]
    print(f"学習 {N_TRAIN}枚 / テスト {N_TEST}枚 のサブセットを使用")
    return TL_CLASSES, tl_test_dl, tl_train_dl


# ============================================================
# Section 3: 学習と比較
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. 3つの方式を同条件で比較する

    同じ2,000枚・同じエポック数で、以下の3つを比べます：

    1. **ゼロから学習**（重みランダム初期化のResNet-18）
    2. **転移学習・特徴抽出**（backbone凍結、fcのみ学習）
    3. **転移学習・ファインチューニング**（全体を小さいlrで学習）

    > ⏱️ 3方式 × 2エポックで **数分〜十数分** かかります（Apple Silicon GPUなら速め）。
    """)
    return


@app.cell
def _(mo):
    tl_epochs = mo.ui.slider(1, 5, step=1, value=2, label="エポック数")
    tl_methods = mo.ui.multiselect(
        options=["ゼロから学習", "特徴抽出", "ファインチューニング"],
        value=["特徴抽出", "ファインチューニング"],
        label="比較する方式",
    )
    tl_btn = mo.ui.run_button(label="比較実験を実行")
    mo.hstack([tl_methods, tl_epochs, tl_btn], justify="start", gap=2)
    return tl_btn, tl_epochs, tl_methods


@app.cell
def _(
    build_model,
    device,
    mo,
    models,
    nn,
    tl_btn,
    tl_epochs,
    tl_methods,
    tl_test_dl,
    tl_train_dl,
    torch,
):
    mo.stop(
        not tl_btn.value,
        mo.md("> ⏸️ 「比較実験を実行」ボタンを押すと学習が始まります"),
    )

    def eval_acc(model):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in tl_test_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                correct += (model(x_batch).argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)
        return correct / total

    def train_one(model, lr, epochs, tag):
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        # requires_grad=False のパラメータはoptimizerに渡さない
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )
        accs = []
        for _epoch in mo.status.progress_bar(
            range(epochs), title=f"学習中: {tag}", subtitle="エポック"
        ):
            model.train()
            for x_batch, y_batch in tl_train_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            accs.append(eval_acc(model))
        return model, accs

    def make_scratch_model():
        """重みランダム初期化のResNet-18（比較用）"""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

    _configs = {
        # 方式名: (モデル生成, 学習率)
        "ゼロから学習": (make_scratch_model, 1e-3),
        "特徴抽出": (lambda: build_model(10, freeze_backbone=True), 1e-3),
        # ファインチューニングは事前学習の知識を壊さないよう小さいlrで
        "ファインチューニング": (lambda: build_model(10, freeze_backbone=False), 1e-4),
    }

    torch.manual_seed(0)
    tl_results = {}
    tl_models = {}
    for _name in tl_methods.value:
        _builder, _lr = _configs[_name]
        _model, _accs = train_one(_builder(), _lr, tl_epochs.value, _name)
        tl_results[_name] = _accs
        tl_models[_name] = _model

    for _name, _accs in tl_results.items():
        print(f"{_name}: 最終テスト正解率 {_accs[-1]:.1%}")
    return tl_models, tl_results


@app.cell
def _(plt, tl_results):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _colors = {"ゼロから学習": "gray", "特徴抽出": "tab:blue",
               "ファインチューニング": "tab:red"}
    for _name, _accs in tl_results.items():
        _ax.plot(range(1, len(_accs) + 1), [_a * 100 for _a in _accs],
                 "o-", lw=2, markersize=8,
                 color=_colors.get(_name), label=_name)
    _ax.axhline(10, color="lightgray", ls="--", label="当てずっぽう（10%）")
    _ax.set_title("同じ2,000枚で学習したときのテスト正解率比較", fontsize=12)
    _ax.set_xlabel("エポック")
    _ax.set_ylabel("テスト正解率（%）")
    _ax.set_ylim(0, 100)
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **観察ポイント**：
    > - 転移学習の2方式は **1エポック目から70〜90%** に到達します。
    >   ImageNetで獲得した特徴抽出器がそのまま効いている証拠です
    > - ゼロから学習は2,000枚では全く追いつけません
    >   （前ノートブックのCNNが健闘したのは1万枚使ったから）
    > - **データが少ないほど転移学習の優位が際立つ** ── これが実務で転移学習が
    >   デフォルトである理由です
    """)
    return


# ============================================================
# Section 4: 予測を見る
# ============================================================
@app.cell
def _(TL_CLASSES, datasets, device, mo, plt, tl_models, torch, v2):
    show_btn = mo.ui.run_button(label="別のサンプルを見る")

    # 最後に学習した方式のモデルで予測する
    _model = list(tl_models.values())[-1]
    _name_used = list(tl_models.keys())[-1]

    _raw_test = datasets.CIFAR10(root="data", train=False, download=False)
    _prep = v2.Compose([
        v2.ToImage(),
        v2.Resize(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _idx = torch.randperm(len(_raw_test))[:8]
    _imgs = [_raw_test[_i][0] for _i in _idx]
    _labels = [_raw_test[_i][1] for _i in _idx]
    _batch = torch.stack([_prep(_im) for _im in _imgs]).to(device)

    _model.eval()
    with torch.no_grad():
        _preds = _model(_batch).argmax(dim=1).cpu()

    _fig, _axes = plt.subplots(1, 8, figsize=(14, 2.6))
    for _i, _ax in enumerate(_axes):
        _ax.imshow(_imgs[_i])
        _ok = _preds[_i].item() == _labels[_i]
        _ax.set_title(
            f"予測: {TL_CLASSES[_preds[_i]]}\n正解: {TL_CLASSES[_labels[_i]]}",
            fontsize=8, color="green" if _ok else "red",
        )
        _ax.axis("off")
    _fig.suptitle(f"「{_name_used}」モデルの予測（緑: 正解, 赤: 不正解）", fontsize=11)
    _fig.tight_layout()
    mo.vstack([show_btn, _fig])
    return


# ============================================================
# まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## まとめ

    | 性質 | ゼロから学習 | 特徴抽出 | ファインチューニング |
    |:---:|:---:|:---:|:---:|
    | 初期重み | ランダム | ImageNet学習済み | ImageNet学習済み |
    | 学習対象 | 全パラメータ | 最終層のみ（約0.05%） | 全パラメータ |
    | 学習率の目安 | 1e-3 | 1e-3 | **1e-4**（知識を壊さないよう小さく） |
    | 必要データ量 | 多い | 少なくてOK | 中くらい |
    | PyTorch | `weights=None` | `requires_grad=False` で凍結 | 凍結なし・小lr |

    **転移学習の3ステップ（覚え方）**：
    1. `models.resnet18(weights=...DEFAULT)` で学習済みモデルを取得
    2. 必要なら凍結、`model.fc` を自分のクラス数に差し替え
    3. 前処理を事前学習時（ImageNet統計・224×224）に揃えて普通に学習

    **次のステップの候補**：
    - データ拡張（`RandomHorizontalFlip` 等）を入れてファインチューニングの精度を伸ばす
    - 自分で集めた画像フォルダを `datasets.ImageFolder` で読み込んで転移学習する
    - Andrej Karpathy「Let's build GPT」でTransformerへ進む
    """)
    return


if __name__ == "__main__":
    app.run()
