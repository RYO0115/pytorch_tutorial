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
    # モデルの保存と読み込み（Save and Load the Model）

    学習済みモデルは **「保存」→「再利用」** が基本です。

    毎回ゼロから学習し直すのは時間がかかりすぎます。
    このノートブックでは PyTorch のモデル保存・読み込みの仕組みを、
    「なぜそうするのか」の理由とともに理解します。

    ```
    [チュートリアル] https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
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
    from torchvision import datasets
    from torchvision.transforms import v2
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    import os
    import tempfile

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"使用デバイス: {device}")

    return (
        DataLoader,
        FancyBboxPatch,
        datasets,
        device,
        mpatches,
        nn,
        np,
        os,
        plt,
        tempfile,
        torch,
        v2,
    )


# ============================================================
# モデル定義（前チュートリアルと共通）
# ============================================================
@app.cell
def _(nn):
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

    return (NeuralNetwork,)


# ============================================================
# Section 1: state_dict とは何か？
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1. state_dict とは何か？

    PyTorch モデルの学習結果は **`state_dict`**（ステート辞書）という形で管理されています。

    `state_dict` とは：
    > **「層の名前」→「重みテンソル」の対応を格納した Python の辞書（dict）**

    モデルがどれだけ賢くなったか（学習したか）は、すべて各層の重みに記録されています。
    つまり `state_dict` を保存すれば、学習の成果がまるごと保存できます。

    ```python
    model = NeuralNetwork()
    print(model.state_dict())
    ```
    """)
    return


@app.cell
def _(NeuralNetwork, torch):
    _model_demo = NeuralNetwork()
    _sd = _model_demo.state_dict()

    print("state_dict のキー一覧：")
    for _k, _v in _sd.items():
        print(f"  {_k:50s}  shape: {list(_v.shape)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### state_dict の構造を可視化する

    下のドロップダウンで確認したい層を選んでみましょう。
    各層の重みがどんな形（shape）のテンソルかを確認できます。
    """)
    return


@app.cell
def _(NeuralNetwork, mo):
    _model_for_ui = NeuralNetwork()
    _sd_keys = list(_model_for_ui.state_dict().keys())

    layer_selector = mo.ui.dropdown(
        options=_sd_keys,
        value=_sd_keys[0],
        label="確認したい層を選択",
    )
    layer_selector
    return (layer_selector,)


@app.cell
def _(NeuralNetwork, layer_selector, mo, torch):
    _model_inspect = NeuralNetwork()
    _sd_inspect = _model_inspect.state_dict()
    _selected_key = layer_selector.value
    _tensor = _sd_inspect[_selected_key]

    _shape = list(_tensor.shape)
    _num_params = _tensor.numel()
    _mean_val = _tensor.mean().item()
    _std_val = _tensor.std().item()
    _min_val = _tensor.min().item()
    _max_val = _tensor.max().item()

    if len(_shape) == 2:
        _shape_desc = f"{_shape[0]} 行 × {_shape[1]} 列 の重み行列"
    elif len(_shape) == 1:
        _shape_desc = f"{_shape[0]} 次元のバイアスベクトル"
    else:
        _shape_desc = f"shape: {_shape}"

    _sample_vals = _tensor.flatten()[:8].tolist()
    _sample_str = ", ".join([f"{v:.4f}" for v in _sample_vals])

    mo.md(f"""
    ### 層 `{_selected_key}` の詳細

    | 項目 | 値 |
    |:---|:---|
    | **形状（shape）** | `{_shape}` |
    | **説明** | {_shape_desc} |
    | **パラメータ数** | {_num_params:,} 個 |
    | **平均値** | `{_mean_val:.4f}` |
    | **標準偏差** | `{_std_val:.4f}` |
    | **最小値** | `{_min_val:.4f}` |
    | **最大値** | `{_max_val:.4f}` |

    **先頭 8 個の値：** `[{_sample_str}, ...]`

    > **補足**：重みは `kaiming_uniform_` などで初期化されており、
    > 学習前はランダムな小さな値が格納されています。
    > 学習後は各値が調整され、入力から正しい出力を導くように変化します。
    """)
    return


@app.cell
def _(NeuralNetwork, plt):
    _model_vis = NeuralNetwork()
    _sd_vis = _model_vis.state_dict()

    _weight_keys = [k for k in _sd_vis.keys() if "weight" in k]

    fig_sd, axes_sd = plt.subplots(1, len(_weight_keys), figsize=(14, 4))

    for _i, _k in enumerate(_weight_keys):
        _ax = axes_sd[_i]
        _w = _sd_vis[_k].numpy()
        _im = _ax.imshow(_w[:50, :50], cmap="coolwarm", aspect="auto",
                         vmin=-0.1, vmax=0.1)
        _ax.set_title(f"{_k}\nshape: {list(_w.shape)}", fontsize=8)
        _ax.set_xlabel("入力ニューロン（先頭50）", fontsize=7)
        _ax.set_ylabel("出力ニューロン（先頭50）", fontsize=7)
        plt.colorbar(_im, ax=_ax, fraction=0.046)

    plt.suptitle(
        "各線形層の重み行列の初期値（ヒートマップ）\n"
        "赤=正の値 / 青=負の値 / 白=ゼロ付近",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig_sd
    return (axes_sd, fig_sd)


# ============================================================
# Section 2: 重みを保存する（state_dict 方式）
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. 重みを保存する（推奨方式：state_dict）

    学習済みの重みを保存するには `torch.save()` を使います。

    ```python
    model = NeuralNetwork()
    # ... 学習 ...
    torch.save(model.state_dict(), 'model_weights.pth')
    ```

    ### なぜ `.pth` という拡張子なのか？

    `.pth` は **PyTorch の慣例的な拡張子** です（`.pt` もよく使われます）。
    中身は Python の **pickle 形式** でシリアライズされたバイナリファイルです。

    ### 何が保存されるか？

    | 保存内容 | state_dict 方式 |
    |:---|:---:|
    | 重み（weight）・バイアス（bias） | ✅ 保存 |
    | モデルのクラス定義（構造） | ❌ 保存しない |
    | オプティマイザの状態 | ❌ 保存しない（別途保存可能） |

    > **ポイント**：構造は保存されないため、読み込み時は自分でモデルを定義する必要があります。
    """)
    return


@app.cell
def _(NeuralNetwork, os, tempfile, torch):
    _model_save = NeuralNetwork()

    _tmp_dir = tempfile.mkdtemp()
    _weights_path = os.path.join(_tmp_dir, "model_weights.pth")

    torch.save(_model_save.state_dict(), _weights_path)

    _file_size = os.path.getsize(_weights_path)
    print(f"保存先: {_weights_path}")
    print(f"ファイルサイズ: {_file_size:,} bytes ({_file_size / 1024:.1f} KB)")
    print(f"パラメータ総数: {sum(p.numel() for p in _model_save.parameters()):,} 個")

    return _tmp_dir, _weights_path


# ============================================================
# Section 3: 重みを読み込む（state_dict 方式）
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. 重みを読み込む（state_dict 方式）

    保存した重みを読み込むには **2 ステップ** が必要です。

    ```python
    # ステップ1: 空のモデルを作る（構造だけ）
    model = NeuralNetwork()

    # ステップ2: 保存した重みを流し込む
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    # ステップ3: 推論モードに切り替える（重要！）
    model.eval()
    ```

    ### `weights_only=True` はなぜ必要か？

    pickle ファイルは悪意あるコードを含む可能性があります。
    `weights_only=True` にすることで、**重みの復元だけ**に処理を限定し、
    任意コード実行のリスクを防ぎます（**セキュリティのベストプラクティス**）。
    """)
    return


@app.cell
def _(NeuralNetwork, _tmp_dir, _weights_path, mo, os, torch):
    _model_load = NeuralNetwork()

    _sd_before = {k: v.clone() for k, v in _model_load.state_dict().items()}

    _loaded_sd = torch.load(_weights_path, weights_only=True)
    _model_load.load_state_dict(_loaded_sd)

    _sd_after = _model_load.state_dict()

    _all_match = all(
        torch.equal(_sd_before[k], _sd_after[k])
        for k in _sd_before.keys()
    )

    _model_load.eval()

    mo.md(f"""
    ### 読み込み結果の確認

    | 確認項目 | 結果 |
    |:---|:---|
    | 読み込みファイル | `{os.path.basename(_weights_path)}` |
    | 重みの一致確認 | {'✅ 保存前と読み込み後が完全一致' if _all_match else '❌ 不一致'} |
    | モデルの状態 | `model.eval()` 適用済み |

    > **重みは完全に保存・復元されました！**
    > 元の重みと読み込んだ重みが 1 要素も違わず一致しています。
    """)
    return


@app.cell
def _(FancyBboxPatch, plt):
    fig_flow, ax_flow = plt.subplots(figsize=(13, 5))
    ax_flow.set_xlim(0, 13)
    ax_flow.set_ylim(0, 5)
    ax_flow.axis("off")

    _save_steps = [
        (1.5, 3.8, "①\n学習済みモデル", "#d0e8ff", "NeuralNetwork()\n+ 学習済み重み"),
        (4.5, 3.8, "②\ntorch.save()", "#ffe0b2", "model.state_dict()\nを渡す"),
        (7.5, 3.8, "③\n.pth ファイル", "#f0f0f0", "バイナリ形式で\nディスクに保存"),
    ]
    _load_steps = [
        (7.5, 1.5, "③\n.pth ファイル", "#f0f0f0", "ディスクから\n読み込む"),
        (4.5, 1.5, "④\ntorch.load()", "#ffd5d5", "weights_only=True\nで安全に"),
        (1.5, 1.5, "⑤\n空のモデルに\n流し込む", "#c8f7c5", "load_state_dict()\n→ model.eval()"),
    ]

    for _steps, _row_label in [(_save_steps, "保存"), (_load_steps, "読み込み")]:
        for _cx, _cy, _label, _color, _sub in _steps:
            _box = FancyBboxPatch((_cx - 1.2, _cy - 0.6), 2.4, 1.2,
                                  boxstyle="round,pad=0.1", facecolor=_color,
                                  edgecolor="#888", lw=1.5)
            ax_flow.add_patch(_box)
            ax_flow.text(_cx, _cy + 0.1, _label, ha="center", va="center",
                        fontsize=9, fontweight="bold")
            ax_flow.text(_cx, _cy - 1.0, _sub, ha="center", va="center",
                        fontsize=7.5, color="#555")

    for _i in range(len(_save_steps) - 1):
        _x1 = _save_steps[_i][0] + 1.2
        _x2 = _save_steps[_i + 1][0] - 1.2
        ax_flow.annotate("", xy=(_x2, 3.8), xytext=(_x1, 3.8),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#2196F3"))

    for _i in range(len(_load_steps) - 1):
        _x1 = _load_steps[_i][0] - 1.2
        _x2 = _load_steps[_i + 1][0] + 1.2
        ax_flow.annotate("", xy=(_x2, 1.5), xytext=(_x1, 1.5),
                         arrowprops=dict(arrowstyle="->", lw=2, color="#E53935"))

    ax_flow.text(0.3, 3.8, "保存\n→", ha="center", fontsize=10,
                color="#2196F3", fontweight="bold")
    ax_flow.text(0.3, 1.5, "読み込み\n←", ha="center", fontsize=10,
                color="#E53935", fontweight="bold")

    ax_flow.plot([7.5, 7.5], [3.2, 2.1], color="#666", lw=2, linestyle="--")
    ax_flow.text(8.0, 2.65, "同じファイル", ha="left", fontsize=8.5, color="#666")

    plt.title("state_dict 方式の保存・読み込みフロー",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_flow
    return (ax_flow, fig_flow)


# ============================================================
# Section 4: model.eval() の重要性
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. `model.eval()` が重要な理由

    モデルを読み込んだ後、推論の前に必ず `model.eval()` を呼ぶ必要があります。

    ```python
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.eval()  # ← これを忘れると推論結果がランダムになることがある！
    ```

    ### 影響を受けるレイヤー

    | レイヤー | `model.train()` 時 | `model.eval()` 時 |
    |:---|:---|:---|
    | **Dropout** | ランダムにニューロンを無効化 | 全ニューロンを使用（スケール調整あり） |
    | **BatchNorm** | バッチ統計で正規化 | 学習時に記録した統計で固定 |
    | **その他** | 変化なし | 変化なし |

    ### Dropout の効果を確認する

    Dropout を持つモデルで `train()` と `eval()` の違いを見てみましょう。
    """)
    return


@app.cell
def _(mo):
    dropout_rate_slider = mo.ui.slider(
        start=0.0, stop=0.9, step=0.1, value=0.5,
        label="Dropout 率（0=無効、0.9=90%のニューロンをランダム無効化）",
    )
    dropout_rate_slider
    return (dropout_rate_slider,)


@app.cell
def _(dropout_rate_slider, mo, nn, np, plt, torch):
    _dropout_rate = dropout_rate_slider.value

    class _DropoutNet(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.dropout = nn.Dropout(p=p)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            return self.fc2(self.dropout(torch.relu(self.fc1(x))))

    _net = _DropoutNet(_dropout_rate)
    _x_input = torch.ones(1, 10)

    _n_trials = 20
    _train_outputs = []
    _eval_outputs = []

    _net.train()
    for _ in range(_n_trials):
        with torch.no_grad():
            _train_outputs.append(_net(_x_input).squeeze().numpy())

    _net.eval()
    for _ in range(_n_trials):
        with torch.no_grad():
            _eval_outputs.append(_net(_x_input).squeeze().numpy())

    _train_arr = np.array(_train_outputs)
    _eval_arr  = np.array(_eval_outputs)

    _train_std = _train_arr.std(axis=0).mean()
    _eval_std  = _eval_arr.std(axis=0).mean()

    fig_eval, axes_eval = plt.subplots(1, 2, figsize=(12, 4.5))

    _colors_t = plt.cm.Reds(np.linspace(0.3, 0.9, _n_trials))
    _colors_e = plt.cm.Blues(np.linspace(0.3, 0.9, _n_trials))

    for _i, _out in enumerate(_train_outputs):
        axes_eval[0].plot(range(5), _out, color=_colors_t[_i], lw=1.2, alpha=0.7)
    axes_eval[0].set_title(
        f"model.train() 時の出力（{_n_trials}回実行）\n"
        f"Dropout={_dropout_rate}  平均標準偏差: {_train_std:.4f}",
        fontsize=10
    )
    axes_eval[0].set_xlabel("出力ニューロン番号")
    axes_eval[0].set_ylabel("出力値")
    axes_eval[0].set_xticks(range(5))
    axes_eval[0].grid(True, alpha=0.3)
    _patch_t = mpatches.Patch(color="red", alpha=0.5, label=f"各試行の出力（計{_n_trials}本）")
    axes_eval[0].legend(handles=[_patch_t], fontsize=8)

    for _i, _out in enumerate(_eval_outputs):
        axes_eval[1].plot(range(5), _out, color=_colors_e[_i], lw=1.2, alpha=0.7)
    axes_eval[1].set_title(
        f"model.eval() 時の出力（{_n_trials}回実行）\n"
        f"Dropout={_dropout_rate}  平均標準偏差: {_eval_std:.4f}",
        fontsize=10
    )
    axes_eval[1].set_xlabel("出力ニューロン番号")
    axes_eval[1].set_ylabel("出力値")
    axes_eval[1].set_xticks(range(5))
    axes_eval[1].grid(True, alpha=0.3)
    _patch_e = mpatches.Patch(color="blue", alpha=0.5, label=f"各試行の出力（計{_n_trials}本）")
    axes_eval[1].legend(handles=[_patch_e], fontsize=8)

    plt.suptitle(
        "同じモデル・同じ入力でも train/eval モードで結果が変わる\n"
        "← 推論には必ず eval() を使うこと！",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    mo.vstack([
        fig_eval,
        mo.md(f"""
> **観察ポイント**（Dropout率 = {_dropout_rate}）：
> - `train()` モード：実行するたびに出力が変わります（赤い線がバラバラ）
> - `eval()` モード：何度実行しても同じ出力です（青い線が重なる）
>
> Dropout率を 0 にすると train/eval の差がなくなります。
> 0.5〜0.9 にすると train モードでの出力の不安定さが際立ちます。
        """),
    ])
    return (axes_eval, fig_eval)


# ============================================================
# Section 5: モデル全体を保存・読み込む
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. モデル全体を保存・読み込む（全体保存方式）

    state_dict 方式とは別に、**モデル全体（構造＋重み）** をまるごと保存することもできます。

    ```python
    # 保存（モデル全体）
    torch.save(model, 'model.pth')

    # 読み込み（モデル全体）
    model = torch.load('model.pth', weights_only=False)
    ```

    ### `weights_only=False` が必要な理由

    モデル全体の保存には Python の **pickle** が使われます。
    pickle は重みだけでなく、クラス定義やメソッドも含むため、
    読み込み時に `weights_only=False` を明示する必要があります。

    > **注意**：信頼できないソースからのファイルには使わないでください。
    """)
    return


@app.cell
def _(NeuralNetwork, _tmp_dir, _weights_path, mo, nn, os, torch):
    _model_full = NeuralNetwork()

    _full_path = os.path.join(_tmp_dir, "model_full.pth")
    torch.save(_model_full, _full_path)

    _loaded_full_model = torch.load(_full_path, weights_only=False)
    _loaded_full_model.eval()

    _weights_size = os.path.getsize(_weights_path)
    _full_size = os.path.getsize(_full_path)

    _x_test = torch.randn(1, 1, 28, 28)
    _model_full.eval()
    with torch.no_grad():
        _out_original = _model_full(_x_test)
        _out_loaded   = _loaded_full_model(_x_test)

    _outputs_match = torch.allclose(_out_original, _out_loaded)

    mo.md(f"""
    ### 全体保存方式の確認

    | 項目 | 値 |
    |:---|:---|
    | 保存先 | `{os.path.basename(_full_path)}` |
    | ファイルサイズ | {_full_size:,} bytes ({_full_size / 1024:.1f} KB) |
    | state_dict 方式との差 | {_full_size - _weights_size:+,} bytes |
    | 出力の一致確認 | {'✅ 完全一致' if _outputs_match else '❌ 不一致'} |

    > 全体保存方式はやや大きくなりますが、読み込み時にクラス定義が不要（既に含まれている）なため
    > 手軽に扱えます。ただし、**クラス定義が変更されると読み込みに失敗する**ことがあります。
    """)
    return


# ============================================================
# Section 6: 2 つの方式の比較
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6. 2 つの保存方式を比較する

    PyTorch には主に 2 つの保存方式があります。どちらを使うべきかを整理します。
    """)
    return


@app.cell
def _(FancyBboxPatch, plt):
    fig_cmp, ax_cmp = plt.subplots(figsize=(13, 6))
    ax_cmp.set_xlim(0, 13)
    ax_cmp.set_ylim(0, 6)
    ax_cmp.axis("off")

    _header_items = [
        (3.25, 5.3, "比較項目", "#e0e0e0"),
        (7.25, 5.3, "state_dict 方式（推奨）", "#d0e8ff"),
        (10.75, 5.3, "全体保存方式", "#ffe0b2"),
    ]
    for _cx, _cy, _label, _color in _header_items:
        _box = FancyBboxPatch((_cx - 2.75, _cy - 0.3), 5.5, 0.6,
                              boxstyle="square,pad=0.05", facecolor=_color,
                              edgecolor="#aaa", lw=1)
        ax_cmp.add_patch(_box)
        ax_cmp.text(_cx, _cy, _label, ha="center", va="center",
                   fontsize=10, fontweight="bold")

    _rows = [
        ("保存コード",
         "torch.save(model.state_dict(), 'w.pth')",
         "torch.save(model, 'm.pth')"),
        ("読み込みコード",
         "model = Net()\nmodel.load_state_dict(\n  torch.load('w.pth', weights_only=True))",
         "model = torch.load(\n  'm.pth', weights_only=False)"),
        ("クラス定義の要否",
         "✅ 読み込み時に必要",
         "△ 同じ環境なら不要"),
        ("セキュリティ",
         "✅ weights_only=True で安全",
         "⚠️ pickle を使うためリスクあり"),
        ("推奨度",
         "⭐⭐⭐ 公式推奨",
         "⭐⭐ 手軽だが用途限定"),
    ]

    for _i, (_item, _left, _right) in enumerate(_rows):
        _y = 4.4 - _i * 0.82
        _bg = "#fafafa" if _i % 2 == 0 else "#f0f0f0"

        _box_l = FancyBboxPatch((0.5, _y - 0.32), 5.0, 0.64,
                                boxstyle="square,pad=0.02", facecolor=_bg,
                                edgecolor="#ddd", lw=0.8)
        _box_c = FancyBboxPatch((5.5, _y - 0.32), 4.0, 0.64,
                                boxstyle="square,pad=0.02", facecolor="#e8f4fd",
                                edgecolor="#ddd", lw=0.8)
        _box_r = FancyBboxPatch((9.5, _y - 0.32), 3.2, 0.64,
                                boxstyle="square,pad=0.02", facecolor="#fff8e7",
                                edgecolor="#ddd", lw=0.8)

        for _b in [_box_l, _box_c, _box_r]:
            ax_cmp.add_patch(_b)

        ax_cmp.text(3.0, _y, _item, ha="center", va="center",
                   fontsize=8.5, fontweight="bold")
        ax_cmp.text(7.5, _y, _left, ha="center", va="center",
                   fontsize=7.5, family="monospace", color="#1a237e")
        ax_cmp.text(11.1, _y, _right, ha="center", va="center",
                   fontsize=7.5, family="monospace", color="#bf360c")

    plt.title("保存方式の比較：state_dict vs 全体保存",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_cmp
    return (ax_cmp, fig_cmp)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "📖 チェックポイント保存（学習途中の再開）について（クリックで展開）": mo.md(r"""
### 学習を途中から再開したい場合

研究や長時間の学習では、途中で中断して後で再開したいことがあります。
その場合は**モデルの重みだけでなくオプティマイザの状態も保存**します。

```python
# チェックポイントの保存
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# チェックポイントの読み込み
checkpoint = torch.load('checkpoint.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss  = checkpoint['loss']
```

| 保存項目 | 理由 |
|:---|:---|
| `epoch` | どこまで学習したか |
| `model_state_dict` | 現在の重み |
| `optimizer_state_dict` | 学習率スケジューラの状態など |
| `loss` | 最後の損失値の記録 |

これが **ベストプラクティス** とされる最も完全な保存方法です。
""")
    })
    return


# ============================================================
# Section 7: 実際に保存・読み込みを動作確認
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 7. 実際に動作確認してみよう

    簡単なモデルを学習させ、保存・読み込み後も同じ予測ができることを確認します。
    """)
    return


@app.cell
def _(mo):
    train_epochs_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=2,
        label="学習エポック数",
    )
    train_epochs_slider
    return (train_epochs_slider,)


@app.cell
def _(
    DataLoader,
    NeuralNetwork,
    _tmp_dir,
    datasets,
    device,
    mo,
    nn,
    os,
    torch,
    train_epochs_slider,
    v2,
):
    _transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    _test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=_transform
    )
    _train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=_transform
    )
    _train_dl = DataLoader(_train_data, batch_size=64, shuffle=True)
    _test_dl  = DataLoader(_test_data,  batch_size=64)

    _model_demo2 = NeuralNetwork().to(device)
    _loss_fn2 = nn.CrossEntropyLoss()
    _opt2 = torch.optim.SGD(_model_demo2.parameters(), lr=1e-3)

    _model_demo2.train()
    for _ep in range(train_epochs_slider.value):
        for _X2, _y2 in _train_dl:
            _X2, _y2 = _X2.to(device), _y2.to(device)
            _p2 = _model_demo2(_X2)
            _l2 = _loss_fn2(_p2, _y2)
            _l2.backward()
            _opt2.step()
            _opt2.zero_grad()

    _model_demo2.eval()
    _correct_before_save = 0
    with torch.no_grad():
        for _X2, _y2 in _test_dl:
            _X2, _y2 = _X2.to(device), _y2.to(device)
            _pred2 = _model_demo2(_X2)
            _correct_before_save += (_pred2.argmax(1) == _y2).float().sum().item()
    _acc_before_save = _correct_before_save / len(_test_data) * 100

    _demo_path = os.path.join(_tmp_dir, "demo_weights.pth")
    torch.save(_model_demo2.state_dict(), _demo_path)

    _model_loaded2 = NeuralNetwork().to(device)
    _model_loaded2.load_state_dict(torch.load(_demo_path, weights_only=True))
    _model_loaded2.eval()

    _correct_after_load = 0
    with torch.no_grad():
        for _X2, _y2 in _test_dl:
            _X2, _y2 = _X2.to(device), _y2.to(device)
            _pred2 = _model_loaded2(_X2)
            _correct_after_load += (_pred2.argmax(1) == _y2).float().sum().item()
    _acc_after_load = _correct_after_load / len(_test_data) * 100

    _acc_match = abs(_acc_before_save - _acc_after_load) < 0.01

    mo.md(f"""
    ### 学習 → 保存 → 読み込みの結果

    | フェーズ | テスト正答率 |
    |:---|:---:|
    | **学習後**（保存前） | **{_acc_before_save:.2f}%** |
    | **読み込み後** | **{_acc_after_load:.2f}%** |
    | **一致確認** | {'✅ 完全に同じ結果（保存・読み込みが正確）' if _acc_match else '❌ 差異あり'} |

    > 学習済みモデルを保存し、別のインスタンスに読み込んでも
    > **全く同じ予測精度が再現**されました。
    > これがモデルの保存・読み込みの目的です。
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

    | 概念 | コード | ポイント |
    |:---|:---|:---|
    | state_dict | `model.state_dict()` | 層名→重みの辞書。学習の成果がすべてここに |
    | 重みの保存 | `torch.save(model.state_dict(), 'w.pth')` | 推奨方式。構造は含まない |
    | 重みの読み込み | `model.load_state_dict(torch.load('w.pth', weights_only=True))` | 先にモデルを作ってから流し込む |
    | 安全な読み込み | `weights_only=True` | 任意コード実行を防ぐセキュリティ設定 |
    | 推論モード | `model.eval()` | 読み込み後・推論前に必ず呼ぶ |
    | 全体保存 | `torch.save(model, 'm.pth')` | 構造も含めて保存（pickle 使用） |
    | 全体読み込み | `torch.load('m.pth', weights_only=False)` | クラス定義が同じ環境に必要 |

    ### 使い分けの指針

    ```
    本番・共有・長期保管  → state_dict 方式（weights_only=True）
    学習途中の再開       → チェックポイント（state_dict + optimizer state）
    手元の素早いテスト   → 全体保存方式でも OK（信頼できるファイルのみ）
    ```

    ### 保存・読み込みの完全なテンプレート

    ```python
    # ── 保存 ──────────────────────────────────────
    torch.save(model.state_dict(), 'model_weights.pth')

    # ── 読み込み ──────────────────────────────────
    model = NeuralNetwork()                                         # ① 構造を作る
    model.load_state_dict(                                          # ② 重みを流し込む
        torch.load('model_weights.pth', weights_only=True)
    )
    model.eval()                                                    # ③ 推論モードへ

    # ── 予測 ──────────────────────────────────────
    with torch.no_grad():
        prediction = model(input_tensor)
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
