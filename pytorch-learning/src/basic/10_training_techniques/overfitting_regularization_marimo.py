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
    # 過学習と正則化：モデルを「暗記」から「理解」へ

    これまでのチュートリアルでは、モデルを学習させて精度が上がることを確認してきました。
    しかし実務で最初にぶつかる壁は **過学習（overfitting）** です。

    > **過学習** = 訓練データを「暗記」してしまい、見たことのないデータ（テストデータ）では
    > 性能が出ない状態

    このノートブックでは、**わざと過学習を起こし**、それを4つの武器で退治します：

    | 対策 | 一言でいうと |
    |:---|:---|
    | **Dropout** | 学習中にニューロンをランダムに無効化し、特定のニューロンへの依存を防ぐ |
    | **Weight Decay** | 重みが大きくなりすぎないよう罰則を加える（L2正則化） |
    | **データ拡張** | 画像を反転・シフトして訓練データを「水増し」する |
    | **早期終了** | テスト性能が悪化し始めたら学習を打ち切り、最良時点のモデルを保存する |
    """)
    return (mo,)


# ============================================================
# インポートとデータ
# ============================================================
@app.cell
def _():
    import copy
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets
    from torchvision.transforms import v2

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"使用デバイス: {device}")
    return DataLoader, Subset, copy, datasets, device, nn, plt, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. 過学習を「わざと」起こす

    過学習が起きやすい条件をそろえます：

    1. **訓練データを少なくする**（60,000枚 → 2,000枚だけ使う）
    2. **モデルを大きくする**（隠れ層512ユニット×2層のMLP ≈ 67万パラメータ）

    「小さいデータ + 大きいモデル」は暗記の温床です。
    2,000枚なら67万パラメータで丸暗記できてしまいます。
    """)
    return


@app.cell
def _(DataLoader, Subset, datasets, torch, v2):
    _plain_tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    # データ拡張あり版：ランダム水平反転 + ランダムシフト
    _aug_tf = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(28, padding=2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    _train_full = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=_plain_tf
    )
    _train_full_aug = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=_aug_tf
    )
    test_ds = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=_plain_tf
    )

    # 同じ2,000枚を「拡張なし」「拡張あり」の2通りで用意する
    _g = torch.Generator().manual_seed(0)
    _indices = torch.randperm(len(_train_full), generator=_g)[:2000].tolist()
    train_small = Subset(_train_full, _indices)
    train_small_aug = Subset(_train_full_aug, _indices)

    test_loader = DataLoader(test_ds, batch_size=512)
    print(f"訓練データ: {len(train_small)} 枚（全体のわずか {len(train_small)/len(_train_full):.1%}）")
    print(f"テストデータ: {len(test_ds)} 枚")
    return test_loader, train_small, train_small_aug


# ============================================================
# Section 2: モデルと実験関数
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. モデルと実験のセットアップ

    Dropoutは `nn.Dropout(p)` を挟むだけです。学習中は各ニューロンの出力を確率 $p$ で0にし、
    残りを $\frac{1}{1-p}$ 倍します（期待値を保つため）。**推論時（`model.eval()`）は何もしません**。

    Weight Decayはオプティマイザの引数1つです：

    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-3)
    ```

    これは損失に $\frac{\lambda}{2}\|W\|^2$ を加えるのと（SGDでは）等価で、
    勾配更新のたびに重みを少しだけ0に引き戻します。
    """)
    return


@app.cell
def _(nn):
    def build_mlp(dropout_p=0.0):
        """隠れ層512×2のMLP。dropout_p > 0 ならDropout層を挟む"""
        layers = [nn.Flatten(), nn.Linear(28 * 28, 512), nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers += [nn.Linear(512, 512), nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(512, 10))
        return nn.Sequential(*layers)

    _n_params = sum(p.numel() for p in build_mlp().parameters())
    print(f"パラメータ数: {_n_params:,}（訓練データ2,000枚に対して過剰）")
    return (build_mlp,)


@app.cell
def _(DataLoader, build_mlp, copy, device, nn, test_loader, torch):
    def evaluate(model):
        """テストデータでの平均損失と正解率"""
        model.eval()
        _loss_fn = nn.CrossEntropyLoss()
        _total_loss, _correct, _n = 0.0, 0, 0
        with torch.no_grad():
            for _x, _y in test_loader:
                _x, _y = _x.to(device), _y.to(device)
                _logits = model(_x)
                _total_loss += _loss_fn(_logits, _y).item() * _y.shape[0]
                _correct += (_logits.argmax(dim=1) == _y).sum().item()
                _n += _y.shape[0]
        return _total_loss / _n, _correct / _n

    def run_experiment(
        train_ds,
        dropout_p=0.0,
        weight_decay=0.0,
        epochs=30,
        patience=None,
        progress=None,
    ):
        """指定した設定で学習し、履歴と最良モデルを返す

        patience を指定すると早期終了：テスト損失が patience エポック連続で
        改善しなければ学習を打ち切る
        """
        torch.manual_seed(0)
        model = build_mlp(dropout_p).to(device)
        _opt = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.9, weight_decay=weight_decay
        )
        _loss_fn = nn.CrossEntropyLoss()
        _loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
        _best_loss, _best_state, _best_epoch, _bad_epochs = float("inf"), None, 0, 0

        for _epoch in range(epochs):
            model.train()
            _total, _correct, _n = 0.0, 0, 0
            for _x, _y in _loader:
                _x, _y = _x.to(device), _y.to(device)
                _logits = model(_x)
                _loss = _loss_fn(_logits, _y)
                _opt.zero_grad()
                _loss.backward()
                _opt.step()
                _total += _loss.item() * _y.shape[0]
                _correct += (_logits.argmax(dim=1) == _y).sum().item()
                _n += _y.shape[0]

            _test_loss, _test_acc = evaluate(model)
            history["train_loss"].append(_total / _n)
            history["train_acc"].append(_correct / _n)
            history["test_loss"].append(_test_loss)
            history["test_acc"].append(_test_acc)

            # チェックポイント：テスト損失が最良なら重みを控えておく
            if _test_loss < _best_loss:
                _best_loss = _test_loss
                _best_state = copy.deepcopy(model.state_dict())
                _best_epoch = _epoch
                _bad_epochs = 0
            else:
                _bad_epochs += 1
                if patience is not None and _bad_epochs >= patience:
                    break  # 早期終了

            if progress is not None:
                progress.update()

        return history, _best_state, _best_epoch
    return evaluate, run_experiment


# ============================================================
# Section 3: 実験の実行
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. 実験：対策なし vs 対策あり

    まず**対策なし**で学習し、次に下で選んだ**対策あり**で学習して比較します。
    注目すべきは「訓練精度とテスト精度の**差**（汎化ギャップ）」です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    use_dropout = mo.ui.checkbox(label="Dropout（p=0.5）", value=True)
    use_wd = mo.ui.checkbox(label="Weight Decay（1e-3）", value=False)
    use_aug = mo.ui.checkbox(label="データ拡張（反転＋シフト）", value=False)
    epochs_slider = mo.ui.slider(10, 60, value=30, step=5, label="エポック数")
    run_btn = mo.ui.run_button(label="実験を実行（対策なし vs 対策あり）")
    mo.vstack(
        [
            mo.md("**「対策あり」に含める対策を選んでください：**"),
            use_dropout,
            use_wd,
            use_aug,
            epochs_slider,
            run_btn,
        ]
    )
    return epochs_slider, run_btn, use_aug, use_dropout, use_wd


@app.cell
def _(
    epochs_slider,
    mo,
    run_btn,
    run_experiment,
    train_small,
    train_small_aug,
    use_aug,
    use_dropout,
    use_wd,
):
    mo.stop(not run_btn.value, mo.md("👆 ボタンを押すと2つの学習が走ります（MPSで1〜2分）"))

    _epochs = epochs_slider.value
    reg_dropout_p = 0.5 if use_dropout.value else 0.0
    with mo.status.progress_bar(total=_epochs * 2, title="学習中...") as _bar:
        # 実験1: 対策なし（ベースライン）
        hist_base, _, _ = run_experiment(train_small, epochs=_epochs, progress=_bar)
        # 実験2: 選んだ対策あり
        hist_reg, best_state, best_epoch = run_experiment(
            train_small_aug if use_aug.value else train_small,
            dropout_p=reg_dropout_p,
            weight_decay=1e-3 if use_wd.value else 0.0,
            epochs=_epochs,
            progress=_bar,
        )

    reg_label = (
        " + ".join(
            _name
            for _flag, _name in [
                (use_dropout.value, "Dropout"),
                (use_wd.value, "WeightDecay"),
                (use_aug.value, "データ拡張"),
            ]
            if _flag
        )
        or "（対策未選択）"
    )
    print(f"対策なし: 最終訓練精度 {hist_base['train_acc'][-1]:.1%} / テスト精度 {hist_base['test_acc'][-1]:.1%}")
    print(f"対策あり（{reg_label}）: 最終訓練精度 {hist_reg['train_acc'][-1]:.1%} / テスト精度 {hist_reg['test_acc'][-1]:.1%}")
    return best_epoch, best_state, hist_base, hist_reg, reg_dropout_p, reg_label


@app.cell
def _(hist_base, hist_reg, plt, reg_label):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    _ax1.plot(hist_base["train_loss"], "C0-", label="訓練（対策なし）")
    _ax1.plot(hist_base["test_loss"], "C0--", label="テスト（対策なし）")
    _ax1.plot(hist_reg["train_loss"], "C1-", label=f"訓練（{reg_label}）")
    _ax1.plot(hist_reg["test_loss"], "C1--", label=f"テスト（{reg_label}）")
    _ax1.set_xlabel("エポック")
    _ax1.set_ylabel("損失")
    _ax1.set_title("損失曲線：実線と破線の開きが「過学習」")
    _ax1.legend(fontsize=8)
    _ax1.grid(alpha=0.3)

    _ax2.plot(hist_base["train_acc"], "C0-", label="訓練（対策なし）")
    _ax2.plot(hist_base["test_acc"], "C0--", label="テスト（対策なし）")
    _ax2.plot(hist_reg["train_acc"], "C1-", label=f"訓練（{reg_label}）")
    _ax2.plot(hist_reg["test_acc"], "C1--", label=f"テスト（{reg_label}）")
    _ax2.set_xlabel("エポック")
    _ax2.set_ylabel("正解率")
    _ax2.set_title("精度曲線")
    _ax2.legend(fontsize=8)
    _ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **読み方のポイント：**

    - 対策なし（青）：訓練精度は100%近くまで上がるのに、テスト損失（青破線）は途中から**上昇に転じる**
      → これが過学習の典型的なサインです
    - 対策あり（オレンジ）：訓練精度の上がり方は遅いが、テスト損失の悪化が抑えられ、
      最終的なテスト精度は高くなる（または同等でギャップが小さい）

    いろいろな組み合わせで再実行して、どの対策が効くか確かめてみてください。
    """)
    return


# ============================================================
# Section 4: 早期終了とチェックポイント
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. 早期終了とチェックポイント：「一番良かった瞬間」を保存する

    損失曲線を見ると、テスト損失には**底**があります。
    それより先に学習を続けるのは、暗記を進めているだけです。

    実務での定番パターン：

    ```python
    if test_loss < best_loss:            # 記録更新なら
        best_loss = test_loss
        torch.save(model.state_dict(), "best.pth")   # チェックポイント保存
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:       # patience 回連続で改善なし
            break                        # → 早期終了
    ```

    上の実験でも裏でこれを実行していました。「最良エポック」のモデルを復元してみます。
    """)
    return


@app.cell
def _(best_epoch, best_state, build_mlp, device, evaluate, hist_reg, reg_dropout_p, torch):
    # 最良エポックの重みをファイルに保存 → 新しいモデルに読み込む（07章の復習）
    torch.save(best_state, "data/best_regularization_model.pth")

    _restored = build_mlp(reg_dropout_p).to(device)
    _restored.load_state_dict(
        torch.load("data/best_regularization_model.pth", weights_only=True)
    )
    _loss, _acc = evaluate(_restored)

    print(f"最良エポック: {best_epoch}（テスト損失が最小だった時点）")
    print(f"最終エポックのテスト精度: {hist_reg['test_acc'][-1]:.2%}")
    print(f"復元した最良モデルのテスト精度: {_acc:.2%}")
    print("→ 'data/best_regularization_model.pth' に保存済み")
    return


# ============================================================
# Section 5: 学習率スケジューラ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. おまけ：学習率スケジューラ

    「最初は大股で、ゴールが近づいたら小股で」歩くために、
    学習率をエポックとともに減らすのが**スケジューラ**です。

    ```python
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    for epoch in range(30):
        train_one_epoch(...)
        scheduler.step()   # エポックの最後に呼ぶ
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sched_dropdown = mo.ui.dropdown(
        options=["StepLR（30エポックごとに1/10）", "CosineAnnealingLR", "OneCycleLR"],
        value="CosineAnnealingLR",
        label="スケジューラを選択",
    )
    sched_dropdown
    return (sched_dropdown,)


@app.cell
def _(build_mlp, plt, sched_dropdown, torch):
    _model = build_mlp()
    _opt = torch.optim.SGD(_model.parameters(), lr=0.1)
    _name = sched_dropdown.value
    _total_epochs = 100

    if _name.startswith("StepLR"):
        _sched = torch.optim.lr_scheduler.StepLR(_opt, step_size=30, gamma=0.1)
    elif _name == "CosineAnnealingLR":
        _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_total_epochs)
    else:
        _sched = torch.optim.lr_scheduler.OneCycleLR(
            _opt, max_lr=0.1, total_steps=_total_epochs
        )

    _lrs = []
    for _ in range(_total_epochs):
        _lrs.append(_opt.param_groups[0]["lr"])
        _sched.step()

    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.plot(_lrs)
    _ax.set_xlabel("エポック")
    _ax.set_ylabel("学習率")
    _ax.set_title(f"{_name} の学習率の推移")
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


# ============================================================
# まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## まとめ

    | 学んだこと | 要点 |
    |:---|:---|
    | 過学習の正体 | 訓練損失↓なのにテスト損失↑。「小さいデータ×大きいモデル」で起きやすい |
    | Dropout | `nn.Dropout(p)` を挟むだけ。`eval()` 時は自動で無効になる |
    | Weight Decay | `optimizer(..., weight_decay=λ)`。重みを0に引き戻すL2罰則 |
    | データ拡張 | `RandomHorizontalFlip` 等。タダで訓練データを増やせる最強の対策 |
    | 早期終了 | テスト損失の底で止める。チェックポイントとセットで使う |
    | スケジューラ | 学習率を徐々に下げる。CosineAnnealingが現代の定番 |

    ### 確認クイズ

    1. Dropout層は推論時（`model.eval()`）に何をするか？
    2. 訓練精度98%・テスト精度85%のモデルと、訓練精度90%・テスト精度88%のモデル、どちらを選ぶべきか？
    3. FashionMNISTで `RandomVerticalFlip`（上下反転）を使うのが不適切なのはなぜか？

    ### 発展課題

    - 訓練データを2,000枚→10,000枚に増やすと過学習はどう変わるか実験する
    - `nn.BatchNorm1d` を追加すると学習がどう変わるか試す
    - CosineAnnealingLRを実際の学習ループに組み込んでテスト精度を比較する
    """)
    return


if __name__ == "__main__":
    app.run()
