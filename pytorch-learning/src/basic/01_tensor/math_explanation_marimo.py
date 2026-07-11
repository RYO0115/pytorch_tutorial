import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # PyTorch Tensor 演算の数学的解説

    [チュートリアル](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
    に登場するベクトル・行列演算と、DL の根幹を成す **偏微分** を視覚的に解説します。
    """)
    return (mo,)


@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D

    plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    return Axes3D, mpatches, np, plt, torch


# ============================================================
# Section 1: テンソルの生成
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1. テンソルの生成

    テンソルの **shape** はその次元構造を表します。

    | 表記 | 意味 | 例 |
    |------|------|----|
    | `(n,)` | n 次元ベクトル | `[1, 2, 3]` |
    | `(m, n)` | m 行 n 列の行列 | 2×3 行列 |

    ```python
    rand_tensor = torch.rand(2, 3)   # 0〜1 の一様乱数
    ones_tensor = torch.ones(2, 3)   # 全要素 1
    zeros_tensor = torch.zeros(2, 3) # 全要素 0
    ```
    """)
    return


@app.cell
def _(np, plt, torch):
    torch.manual_seed(0)
    _rand = torch.rand(2, 3).numpy()
    _ones = torch.ones(2, 3).numpy()
    _zeros = torch.zeros(2, 3).numpy()

    fig1, axes1 = plt.subplots(1, 3, figsize=(11, 3))

    for _ax, _mat, _title, _cmap in zip(
        axes1,
        [_rand, _ones, _zeros],
        ["torch.rand(2,3)\n（0〜1 の乱数）", "torch.ones(2,3)\n（全要素 = 1）", "torch.zeros(2,3)\n（全要素 = 0）"],
        ["YlOrRd", "Blues", "Greys"],
    ):
        _im = _ax.imshow(_mat, cmap=_cmap, vmin=0, vmax=1)
        fig1.colorbar(_im, ax=_ax, shrink=0.8)
        for _r in range(2):
            for _c in range(3):
                _ax.text(_c, _r, f"{_mat[_r,_c]:.2f}", ha="center", va="center",
                         fontsize=11, fontweight="bold")
        _ax.set_title(_title, fontsize=10)
        _ax.set_xticks(range(3))
        _ax.set_yticks(range(2))
        _ax.set_xticklabels([f"[:,{_i}]" for _i in range(3)], fontsize=9)
        _ax.set_yticklabels([f"[{_i},:]" for _i in range(2)], fontsize=9)

    plt.suptitle("テンソルの生成：shape (2, 3) の例", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1
    return axes1, fig1


# ============================================================
# Section 2: インデックスとスライス
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. インデックスとスライス

    ```python
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0   # 2列目をすべて0に
    tensor[1, 2] = 0   # (1,2)要素を0に
    ```

    スライダーで「どこを取り出すか」を変えると、対応する行・列がハイライトされます。
    """)
    return


@app.cell
def _(mo):
    slice_mode = mo.ui.dropdown(
        options={"行を選択 tensor[i]": "row", "列を選択 tensor[:,j]": "col", "要素を選択 tensor[i,j]": "elem"},
        value="行を選択 tensor[i]",
        label="スライスの種類",
    )
    slice_idx_i = mo.ui.slider(start=0, stop=3, step=1, value=0, label="行 i")
    slice_idx_j = mo.ui.slider(start=0, stop=3, step=1, value=1, label="列 j")
    mo.hstack([slice_mode, slice_idx_i, slice_idx_j])
    return slice_idx_i, slice_idx_j, slice_mode


@app.cell
def _(np, plt, slice_idx_i, slice_idx_j, slice_mode, torch):
    _A = torch.ones(4, 4)
    _A[:, 1] = 0
    _A[1, 2] = 0
    _A_np = _A.numpy()

    _mode = slice_mode.value
    _i = slice_idx_i.value
    _j = slice_idx_j.value

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1]})

    # 左：行列のヒートマップ（ハイライト付き）
    _ax_mat = axes2[0]
    _ax_mat.imshow(_A_np, cmap="Blues", vmin=-0.3, vmax=1.3, alpha=0.3)
    for _r in range(4):
        for _c in range(4):
            _highlight = (
                (_mode == "row" and _r == _i) or
                (_mode == "col" and _c == _j) or
                (_mode == "elem" and _r == _i and _c == _j)
            )
            _color = "orangered" if _highlight else "#333"
            _bg = "yellow" if _highlight else "white"
            _ax_mat.add_patch(plt.Rectangle((_c - 0.5, _r - 0.5), 1, 1,
                                            facecolor=_bg, alpha=0.5 if _highlight else 0.0,
                                            edgecolor="gray", lw=0.5))
            _ax_mat.text(_c, _r, f"{int(_A_np[_r,_c])}", ha="center", va="center",
                         fontsize=14, fontweight="bold", color=_color)
    _ax_mat.set_xticks(range(4))
    _ax_mat.set_yticks(range(4))
    _ax_mat.set_xticklabels([f"col{_c}" for _c in range(4)])
    _ax_mat.set_yticklabels([f"row{_r}" for _r in range(4)])
    _ax_mat.set_title("tensor（4×4）", fontsize=11)

    # 右：取り出された値
    _ax_val = axes2[1]
    if _mode == "row":
        _extracted = _A_np[_i, :]
        _code = f"tensor[{_i}]"
        _ax_val.bar(range(4), _extracted, color="steelblue", edgecolor="white")
        _ax_val.set_xticks(range(4))
        _ax_val.set_xticklabels([f"[:,{_c}]" for _c in range(4)])
        _ax_val.set_ylabel("値")
        _ax_val.set_ylim(-0.1, 1.3)
        for _c, _v in enumerate(_extracted):
            _ax_val.text(_c, _v + 0.05, str(int(_v)), ha="center", fontsize=12, fontweight="bold")
    elif _mode == "col":
        _extracted = _A_np[:, _j]
        _code = f"tensor[:, {_j}]"
        _ax_val.barh(range(4), _extracted, color="orangered", edgecolor="white")
        _ax_val.set_yticks(range(4))
        _ax_val.set_yticklabels([f"[{_r},:]" for _r in range(4)])
        _ax_val.set_xlabel("値")
        _ax_val.set_xlim(-0.1, 1.3)
        _ax_val.invert_yaxis()
        for _r, _v in enumerate(_extracted):
            _ax_val.text(_v + 0.05, _r, str(int(_v)), va="center", fontsize=12, fontweight="bold")
    else:
        _v = _A_np[_i, _j]
        _code = f"tensor[{_i}, {_j}]"
        _ax_val.text(0.5, 0.5, f"tensor[{_i},{_j}] = {int(_v)}",
                     ha="center", va="center", fontsize=20, fontweight="bold",
                     transform=_ax_val.transAxes,
                     color="steelblue" if _v == 1 else "orangered")
        _ax_val.axis("off")

    _ax_val.set_title(f"取り出した値：{_code}", fontsize=11)
    _ax_val.grid(True, alpha=0.3, axis="y" if _mode == "row" else "x") if _mode != "elem" else None

    plt.suptitle("インデックスとスライスの可視化", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2
    return axes2, fig2


# ============================================================
# Section 3: テンソルの連結
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. テンソルの連結（Concatenation）

    ```python
    t1 = torch.cat([tensor, tensor, tensor], dim=1)  # 列方向（横）に連結
    ```

    `dim=0` → 行方向（縦）に連結、`dim=1` → 列方向（横）に連結
    """)
    return


@app.cell
def _(mo):
    cat_dim = mo.ui.dropdown(
        options={"dim=0（行方向・縦）": 0, "dim=1（列方向・横）": 1},
        value="dim=1（列方向・横）",
        label="連結方向",
    )
    cat_dim
    return (cat_dim,)


@app.cell
def _(cat_dim, np, plt, torch):
    _A = torch.ones(4, 4)
    _A[:, 1] = 0
    _A[1, 2] = 0
    _dim = cat_dim.value
    _cat = torch.cat([_A, _A, _A], dim=_dim).numpy()
    _A_np = _A.numpy()

    _colors = ["#aec6e8", "#ffd5a8", "#b8f0b8"]
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4),
                                gridspec_kw={"width_ratios": [1, 3 if _dim == 1 else 1]})

    _ax_orig = axes3[0]
    _ax_orig.imshow(_A_np, cmap="Blues", vmin=-0.3, vmax=1.3, alpha=0.6)
    for _r in range(4):
        for _c in range(4):
            _ax_orig.text(_c, _r, str(int(_A_np[_r, _c])), ha="center", va="center", fontsize=11)
    _ax_orig.set_title(f"元の行列 A\nshape {tuple(_A.shape)}", fontsize=10)

    _ax_cat = axes3[1]
    _ax_cat.imshow(np.zeros_like(_cat), cmap="Greys", alpha=0.0)
    for _k in range(3):
        if _dim == 1:
            _offset_c = _k * 4
            _offset_r = 0
        else:
            _offset_c = 0
            _offset_r = _k * 4
        for _r in range(4):
            for _c in range(4):
                _ax_cat.add_patch(plt.Rectangle(
                    (_offset_c + _c - 0.5, _offset_r + _r - 0.5), 1, 1,
                    facecolor=_colors[_k], alpha=0.5, edgecolor="white", lw=0.5))
                _ax_cat.text(_offset_c + _c, _offset_r + _r,
                             str(int(_cat[_offset_r + _r, _offset_c + _c])),
                             ha="center", va="center", fontsize=10)

    _ax_cat.set_xlim(-0.5, _cat.shape[1] - 0.5)
    _ax_cat.set_ylim(_cat.shape[0] - 0.5, -0.5)
    _ax_cat.set_title(f"連結後  dim={_dim}\nshape {tuple(_cat.shape)}", fontsize=10)
    _ax_cat.set_xticks([])
    _ax_cat.set_yticks([])

    _handles = [mpatches.Patch(facecolor=_colors[_k], alpha=0.6, label=f"A のコピー {_k+1}") for _k in range(3)]
    _ax_cat.legend(handles=_handles, loc="upper right", fontsize=9)

    plt.suptitle("torch.cat によるテンソルの連結", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3
    return axes3, fig3


# ============================================================
# Section 4: 行列積
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. 行列積（Matrix Multiplication）

    $$C_{ij} = \\sum_{k=0}^{n-1} A_{ik} \\cdot (A^\\top)_{kj}$$

    **スライダー**で行 $i$ と列 $j$ を選ぶと、$C[i,j]$ がどの要素の掛け算から計算されるかを確認できます。
    """)
    return


@app.cell
def _(mo):
    mm_i = mo.ui.slider(start=0, stop=3, step=1, value=0, label="行 i（A の第i行）")
    mm_j = mo.ui.slider(start=0, stop=3, step=1, value=0, label="列 j（A^T の第j列）")
    mo.hstack([mm_i, mm_j])
    return mm_i, mm_j


@app.cell
def _(mm_i, mm_j, np, plt, torch):
    _A = torch.ones(4, 4)
    _A[:, 1] = 0
    _A[1, 2] = 0
    _A_np = _A.numpy()
    _AT_np = _A_np.T
    _C_np = _A_np @ _AT_np

    _i = mm_i.value
    _j = mm_j.value
    _row = _A_np[_i]
    _col = _AT_np[:, _j]
    _products = _row * _col
    _c_ij = int(_C_np[_i, _j])

    fig4, axes4 = plt.subplots(1, 4, figsize=(15, 4.5),
                                gridspec_kw={"width_ratios": [1, 1, 1.2, 1]})

    def _draw_matrix(ax, mat, title, hi_row=None, hi_col=None, hi_cell=None, cmap="Blues"):
        _n_r, _n_c = mat.shape
        ax.imshow(mat, cmap=cmap, vmin=-0.3, vmax=max(mat.max(), 1) + 0.3, alpha=0.25)
        for _r in range(_n_r):
            for _c in range(_n_c):
                _hl = (hi_row is not None and _r == hi_row) or \
                      (hi_col is not None and _c == hi_col) or \
                      (hi_cell is not None and (_r, _c) == hi_cell)
                _bg = "yellow" if _hl else "white"
                _fw = "bold" if _hl else "normal"
                _fc = "orangered" if _hl else "#444"
                ax.add_patch(plt.Rectangle((_c - 0.5, _r - 0.5), 1, 1,
                                           facecolor=_bg, alpha=0.55 if _hl else 0.0,
                                           edgecolor="gray", lw=0.5))
                ax.text(_c, _r, f"{int(mat[_r,_c])}", ha="center", va="center",
                        fontsize=12, fontweight=_fw, color=_fc)
        ax.set_xticks(range(_n_c))
        ax.set_yticks(range(_n_r))
        ax.set_title(title, fontsize=10)

    _draw_matrix(axes4[0], _A_np, f"A（第{_i}行をハイライト）", hi_row=_i)
    _draw_matrix(axes4[1], _AT_np, f"A^T（第{_j}列をハイライト）", hi_col=_j)
    _draw_matrix(axes4[3], _C_np, f"C = A @ A^T\nC[{_i},{_j}] = {_c_ij}", hi_cell=(_i, _j), cmap="Reds")

    # 途中計算の棒グラフ
    _ax_calc = axes4[2]
    _x_pos = np.arange(4)
    _bars = _ax_calc.bar(_x_pos, _products,
                         color=["steelblue" if _p > 0 else "lightgray" for _p in _products],
                         edgecolor="white")
    _ax_calc.axhline(0, color="black", lw=0.8)
    for _k, (_b, _v) in enumerate(zip(_bars, _products)):
        _ax_calc.text(_k, _v + 0.03, f"{int(_row[_k])}×{int(_col[_k])}={int(_v)}",
                      ha="center", fontsize=8.5, fontweight="bold")
    _ax_calc.set_xticks(_x_pos)
    _ax_calc.set_xticklabels([f"k={_k}" for _k in range(4)])
    _ax_calc.set_ylim(-0.2, 1.5)
    _ax_calc.set_title(f"要素積（A[{_i},k] × A^T[k,{_j}]）\n合計 = {_c_ij}", fontsize=10)
    _ax_calc.grid(True, axis="y", alpha=0.3)
    _ax_calc.text(0.5, 0.95,
                  f"C[{_i},{_j}] = {' + '.join([str(int(_p)) for _p in _products])} = {_c_ij}",
                  transform=_ax_calc.transAxes, ha="center", va="top",
                  fontsize=10, fontweight="bold", color="orangered",
                  bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.suptitle(f"行列積のしくみ：C[{_i},{_j}] = A の第{_i}行 · A^T の第{_j}列", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig4
    return axes4, fig4


# ============================================================
# Section 5: 要素積
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 5. 要素積（Element-wise Product / Hadamard Product）

    $$Z_{ij} = A_{ij} \\cdot B_{ij}$$

    **同じ位置** の要素同士だけを掛け合わせます。行列積と違い shape は変わりません。
    """)
    return


@app.cell
def _(plt, torch):
    _A = torch.ones(4, 4)
    _A[:, 1] = 0
    _A[1, 2] = 0
    _A_np = _A.numpy()
    _Z_np = (_A * _A).numpy()

    fig5, axes5 = plt.subplots(1, 5, figsize=(15, 3.5),
                                gridspec_kw={"width_ratios": [1, 0.3, 1, 0.3, 1]})

    def _draw_small_mat(ax, mat, title, cmap="Blues"):
        ax.imshow(mat, cmap=cmap, vmin=-0.3, vmax=1.3, alpha=0.4)
        for _r in range(4):
            for _c in range(4):
                ax.text(_c, _r, str(int(mat[_r, _c])), ha="center", va="center",
                        fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    _draw_small_mat(axes5[0], _A_np, "A")
    axes5[1].text(0.5, 0.5, "⊙", ha="center", va="center", fontsize=24, transform=axes5[1].transAxes)
    axes5[1].axis("off")
    _draw_small_mat(axes5[2], _A_np, "A（同じ行列）")
    axes5[3].text(0.5, 0.5, "=", ha="center", va="center", fontsize=24, transform=axes5[3].transAxes)
    axes5[3].axis("off")
    _draw_small_mat(axes5[4], _Z_np, "Z = A ⊙ A\n（= A.pow(2)）", cmap="YlOrRd")

    # 対応する要素を矢印で強調
    for _r in range(4):
        for _c in range(4):
            if _A_np[_r, _c] != 0:
                axes5[0].add_patch(plt.Rectangle((_c - 0.5, _r - 0.5), 1, 1,
                                                   facecolor="yellow", alpha=0.3, edgecolor="orange", lw=1))
                axes5[2].add_patch(plt.Rectangle((_c - 0.5, _r - 0.5), 1, 1,
                                                   facecolor="yellow", alpha=0.3, edgecolor="orange", lw=1))
                axes5[4].add_patch(plt.Rectangle((_c - 0.5, _r - 0.5), 1, 1,
                                                   facecolor="yellow", alpha=0.3, edgecolor="orange", lw=1))

    plt.suptitle("要素積：同じ位置の要素同士を掛け合わせる（0²=0, 1²=1）", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig5
    return axes5, fig5


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 行列積との比較

    | | 行列積 `@` | 要素積 `*` |
    |:---|:---:|:---:|
    | $Z_{ij}$ の計算に使う要素 | A の第$i$行 **全体** × B の第$j$列 **全体** を足す | A の $(i,j)$ × B の $(i,j)$ だけ |
    | 必要な shape | A: $(m,n)$、B: $(n,p)$ | A と B が **同じ shape** |
    | 出力 shape | $(m, p)$ | $(m, n)$（入力と同じ） |
    """)
    return


# ============================================================
# Section 6: べき乗
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 6. べき乗（Element-wise Power）

    ```python
    tensor.pow(2)  # 各要素を2乗
    ```

    $$Z_{ij} = A_{ij}^n$$

    `tensor.pow(2)` と `tensor * tensor` は数値的に同じ結果です（$A^{\\odot 2} = A \\odot A$）。
    """)
    return


@app.cell
def _(mo):
    pow_n = mo.ui.slider(start=1, stop=5, step=1, value=2, label="べき乗 n")
    pow_n
    return (pow_n,)


@app.cell
def _(np, plt, pow_n, torch):
    _A = torch.ones(4, 4)
    _A[:, 1] = 0
    _A[1, 2] = 0
    _A_np = _A.numpy()
    _n = pow_n.value
    _Z_pow = (_A ** _n).numpy()

    fig6, axes6 = plt.subplots(1, 3, figsize=(11, 3.5),
                                gridspec_kw={"width_ratios": [1, 0.3, 1]})

    _im_a = axes6[0].imshow(_A_np, cmap="Blues", vmin=0, vmax=1.3, alpha=0.4)
    fig6.colorbar(_im_a, ax=axes6[0], shrink=0.8)
    for _r in range(4):
        for _c in range(4):
            axes6[0].text(_c, _r, str(int(_A_np[_r, _c])), ha="center", va="center", fontsize=13, fontweight="bold")
    axes6[0].set_title("A", fontsize=11)
    axes6[0].set_xticks([])
    axes6[0].set_yticks([])

    axes6[1].text(0.5, 0.5, f".pow({_n})\n→", ha="center", va="center", fontsize=14, transform=axes6[1].transAxes)
    axes6[1].axis("off")

    _im_z = axes6[2].imshow(_Z_pow, cmap="YlOrRd", vmin=0, vmax=1.3, alpha=0.4)
    fig6.colorbar(_im_z, ax=axes6[2], shrink=0.8)
    for _r in range(4):
        for _c in range(4):
            _v = _Z_pow[_r, _c]
            axes6[2].text(_c, _r, f"{_v:.0f}", ha="center", va="center", fontsize=13, fontweight="bold")
    axes6[2].set_title(f"A.pow({_n})  = 各要素の {_n} 乗", fontsize=11)
    axes6[2].set_xticks([])
    axes6[2].set_yticks([])

    _x_line = np.linspace(0, 1, 100)
    _inset = axes6[2].inset_axes([0.55, 0.55, 0.42, 0.42])
    _inset.plot(_x_line, _x_line ** _n, color="orangered", lw=2)
    _inset.set_xlim(0, 1)
    _inset.set_xlabel("x", fontsize=7)
    _inset.set_ylabel(f"x^{_n}", fontsize=7)
    _inset.set_title(f"y=x^{_n}", fontsize=7)
    _inset.grid(True, alpha=0.3)

    plt.suptitle(f"べき乗：各要素を {_n} 乗する（スライダーで n を変更）", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig6
    return axes6, fig6


# ============================================================
# Section 7: 偏微分
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 7. 偏微分（Partial Derivative）

    > **DL のパラメータ更新の根拠となる最重要概念です。**

    ### 定義

    複数の変数を持つ関数で、**注目する変数以外をすべて定数とみなして微分する**操作。

    $$\\frac{\\partial f}{\\partial w} \\approx \\frac{f(w+\\varepsilon, b) - f(w, b)}{\\varepsilon}$$

    ### 具体例：$f(w, b) = w^2 + wb + b^2$

    $$\\frac{\\partial f}{\\partial w} = 2w + b \\qquad \\frac{\\partial f}{\\partial b} = w + 2b$$

    スライダーで $(w, b)$ の値を変えると、偏微分の値と「勾配の向き」がリアルタイムで変わります。
    """)
    return


@app.cell
def _(mo):
    pd_w = mo.ui.slider(start=-3.0, stop=3.0, step=0.25, value=1.0, label="w の値", full_width=True)
    pd_b = mo.ui.slider(start=-3.0, stop=3.0, step=0.25, value=1.0, label="b の値", full_width=True)
    mo.vstack([pd_w, pd_b])
    return pd_b, pd_w


@app.cell
def _(np, pd_b, pd_w, plt):
    _w0 = pd_w.value
    _b0 = pd_b.value

    def _f(w, b): return w**2 + w*b + b**2
    def _df_dw(w, b): return 2*w + b
    def _df_db(w, b): return w + 2*b

    _W = np.linspace(-3, 3, 80)
    _B = np.linspace(-3, 3, 80)
    _WW, _BB = np.meshgrid(_W, _B)
    _FF = _f(_WW, _BB)

    fig7, axes7 = plt.subplots(1, 3, figsize=(15, 5))

    # --- 左：等高線 + 勾配ベクトル ---
    _ax_cnt = axes7[0]
    _cnt = _ax_cnt.contourf(_WW, _BB, _FF, levels=20, cmap="YlOrRd", alpha=0.8)
    plt.colorbar(_cnt, ax=_ax_cnt, shrink=0.8)
    _ax_cnt.contour(_WW, _BB, _FF, levels=10, colors="white", alpha=0.3, linewidths=0.8)

    _gw = _df_dw(_w0, _b0)
    _gb = _df_db(_w0, _b0)
    _scale = 0.3
    _ax_cnt.annotate("", xy=(_w0 + _gw*_scale, _b0 + _gb*_scale), xytext=(_w0, _b0),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=2.5))
    _ax_cnt.scatter([_w0], [_b0], color="blue", s=80, zorder=5)
    _ax_cnt.set_xlabel("w")
    _ax_cnt.set_ylabel("b")
    _ax_cnt.set_title(
        f"f(w,b) の等高線\n現在地 w={_w0}, b={_b0}\n勾配ベクトル（青矢印）", fontsize=10)

    # --- 中央：w 方向のスライス（∂f/∂w の可視化）---
    _ax_w = axes7[1]
    _w_line = np.linspace(-3, 3, 200)
    _f_along_w = _f(_w_line, _b0)
    _ax_w.plot(_w_line, _f_along_w, color="steelblue", lw=2, label=f"f(w, b={_b0})")
    _ax_w.scatter([_w0], [_f(_w0, _b0)], color="red", s=80, zorder=5)

    _tangent_x = np.linspace(_w0 - 1.2, _w0 + 1.2, 50)
    _tangent_y = _f(_w0, _b0) + _gw * (_tangent_x - _w0)
    _ax_w.plot(_tangent_x, _tangent_y, color="orangered", lw=2, linestyle="--",
               label=f"接線（傾き = ∂f/∂w = {_gw:.2f}）")
    _ax_w.set_xlim(-3, 3)
    _ax_w.set_xlabel("w")
    _ax_w.set_ylabel("f(w, b)")
    _ax_w.set_title(f"w 方向のスライス（b={_b0} 固定）\n∂f/∂w = 2w+b = {_gw:.2f}", fontsize=10)
    _ax_w.legend(fontsize=8)
    _ax_w.grid(True, alpha=0.3)

    # --- 右：b 方向のスライス（∂f/∂b の可視化）---
    _ax_b = axes7[2]
    _b_line = np.linspace(-3, 3, 200)
    _f_along_b = _f(_w0, _b_line)
    _ax_b.plot(_b_line, _f_along_b, color="green", lw=2, label=f"f(w={_w0}, b)")
    _ax_b.scatter([_b0], [_f(_w0, _b0)], color="red", s=80, zorder=5)

    _tangent_bx = np.linspace(_b0 - 1.2, _b0 + 1.2, 50)
    _tangent_by = _f(_w0, _b0) + _gb * (_tangent_bx - _b0)
    _ax_b.plot(_tangent_bx, _tangent_by, color="orangered", lw=2, linestyle="--",
               label=f"接線（傾き = ∂f/∂b = {_gb:.2f}）")
    _ax_b.set_xlim(-3, 3)
    _ax_b.set_xlabel("b")
    _ax_b.set_ylabel("f(w, b)")
    _ax_b.set_title(f"b 方向のスライス（w={_w0} 固定）\n∂f/∂b = w+2b = {_gb:.2f}", fontsize=10)
    _ax_b.legend(fontsize=8)
    _ax_b.grid(True, alpha=0.3)

    plt.suptitle(
        f"f(w,b) = w² + wb + b²  の偏微分  ｜  ∂f/∂w = {_gw:.2f}、∂f/∂b = {_gb:.2f}",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig7
    return axes7, fig7


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### DL での偏微分の役割：勾配降下法

    偏微分を使ってパラメータを更新するアルゴリズムを **勾配降下法** といいます。

    $$w \\leftarrow w - \\alpha \\cdot \\frac{\\partial L}{\\partial w}, \\qquad b \\leftarrow b - \\alpha \\cdot \\frac{\\partial L}{\\partial b}$$

    スライダーで学習率と初期値を変えて、勾配降下の軌跡を観察してください。
    """)
    return


@app.cell
def _(mo):
    gd_lr = mo.ui.slider(start=0.05, stop=0.5, step=0.05, value=0.2, label="学習率 α")
    gd_steps = mo.ui.slider(start=1, stop=30, step=1, value=10, label="ステップ数")
    gd_w0 = mo.ui.slider(start=-2.5, stop=2.5, step=0.5, value=2.0, label="初期値 w₀")
    gd_b0 = mo.ui.slider(start=-2.5, stop=2.5, step=0.5, value=-2.0, label="初期値 b₀")
    mo.hstack([mo.vstack([gd_lr, gd_steps]), mo.vstack([gd_w0, gd_b0])])
    return gd_b0, gd_lr, gd_steps, gd_w0


@app.cell
def _(gd_b0, gd_lr, gd_steps, gd_w0, np, plt):
    def _f_gd(w, b): return w**2 + w*b + b**2
    def _grad_w(w, b): return 2*w + b
    def _grad_b(w, b): return w + 2*b

    _lr = gd_lr.value
    _n_steps = gd_steps.value
    _wi, _bi = gd_w0.value, gd_b0.value

    _path_w = [_wi]
    _path_b = [_bi]
    _path_f = [_f_gd(_wi, _bi)]

    for _ in range(_n_steps):
        _gw_step = _grad_w(_wi, _bi)
        _gb_step = _grad_b(_wi, _bi)
        _wi = _wi - _lr * _gw_step
        _bi = _bi - _lr * _gb_step
        _path_w.append(_wi)
        _path_b.append(_bi)
        _path_f.append(_f_gd(_wi, _bi))

    _W_gd = np.linspace(-3.5, 3.5, 100)
    _B_gd = np.linspace(-3.5, 3.5, 100)
    _WW_gd, _BB_gd = np.meshgrid(_W_gd, _B_gd)
    _FF_gd = _f_gd(_WW_gd, _BB_gd)

    fig8, axes8 = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左：等高線上の軌跡 ---
    _ax_traj = axes8[0]
    _cnt2 = _ax_traj.contourf(_WW_gd, _BB_gd, _FF_gd, levels=25, cmap="YlOrRd", alpha=0.8)
    plt.colorbar(_cnt2, ax=_ax_traj, shrink=0.8, label="f(w,b)")
    _ax_traj.contour(_WW_gd, _BB_gd, _FF_gd, levels=12, colors="white", alpha=0.3, linewidths=0.8)

    _ax_traj.plot(_path_w, _path_b, "o-", color="blue", lw=2, ms=6, zorder=5)
    _ax_traj.plot(_path_w[0], _path_b[0], "s", color="lime", ms=12, zorder=6, label="開始点")
    _ax_traj.plot(_path_w[-1], _path_b[-1], "*", color="blue", ms=14, zorder=6, label=f"終点（step {_n_steps}）")
    _ax_traj.plot(0, 0, "+", color="black", ms=15, mew=3, zorder=7, label="最小値 (0,0)")
    _ax_traj.set_xlabel("w")
    _ax_traj.set_ylabel("b")
    _ax_traj.set_title(f"勾配降下の軌跡（α={_lr}、{_n_steps}ステップ）", fontsize=10)
    _ax_traj.legend(fontsize=9)

    # --- 右：損失の推移 ---
    _ax_loss = axes8[1]
    _ax_loss.plot(range(_n_steps + 1), _path_f, "o-", color="orangered", lw=2, ms=6)
    _ax_loss.set_xlabel("ステップ数")
    _ax_loss.set_ylabel("f(w, b)（損失）")
    _ax_loss.set_title(
        f"損失の推移\n開始: {_path_f[0]:.3f} → 終了: {_path_f[-1]:.4f}", fontsize=10)
    _ax_loss.grid(True, alpha=0.3)
    for _s in range(0, _n_steps + 1, max(1, _n_steps // 5)):
        _ax_loss.annotate(f"step{_s}\n{_path_f[_s]:.3f}",
                          xy=(_s, _path_f[_s]),
                          xytext=(_s + 0.5, _path_f[_s] + (_path_f[0] - _path_f[-1]) * 0.07),
                          fontsize=7.5, color="steelblue")

    plt.suptitle(
        f"勾配降下法：偏微分を使ってパラメータを最小値に向けて更新する",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig8
    return axes8, fig8


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### PyTorch での偏微分：`loss.backward()` との対応

    `back_propagation.py` の1層ネットワークでは：

    ```python
    z = torch.matmul(x, w) + b    # z = xW + b
    loss = BCE(z, y)               # 損失

    loss.backward()   # ← 自動で ∂loss/∂w と ∂loss/∂b を計算

    print(w.grad)     # ∂loss/∂w（shape: 5×3 の行列）
    print(b.grad)     # ∂loss/∂b（shape: 3 のベクトル）
    ```

    | PyTorch | 数学 | 意味 |
    |:---|:---|:---|
    | `w.grad` | $\\nabla_w L$ | w の各要素に対する損失の偏微分 |
    | `b.grad` | $\\nabla_b L$ | b の各要素に対する損失の偏微分 |
    | `loss.backward()` | 連鎖律 | 計算グラフを逆向きに辿って全偏微分を自動計算 |

    ### 学習ループにおける偏微分の位置づけ

    ```
    ① 順伝播：入力 → ネットワーク → 損失 L を計算（計算グラフを構築）
    ② 逆伝播：loss.backward() → 連鎖律で ∂L/∂w, ∂L/∂b を自動計算
    ③ 更新  ：w ← w - α * ∂L/∂w　／　b ← b - α * ∂L/∂b
    ④ リセット：w.grad.zero_()
    ⑤ ① に戻る
    ```

    > **偏微分は「どこに向かってパラメータを動かすか」の羅針盤です。**
    > PyTorch の `autograd` はこの羅針盤を自動で計算してくれる仕組みです。
    """)
    return


# ============================================================
# Section: 収束条件 — 偏微分がゼロになると学習完了
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 学習の完了条件：偏微分がゼロに近づいたとき

    #### なぜ偏微分 = 0 が「学習完了」の合図か？

    損失関数 $f(w, b) = w^2 + wb + b^2$ を例にとります。

    $$\frac{\partial f}{\partial w} = 2w + b, \qquad \frac{\partial f}{\partial b} = w + 2b$$

    **偏微分の意味**：その変数の方向に少し動いたとき、損失がどれだけ増えるかの傾きです。

    | 偏微分の値 | 意味 | パラメータの動き |
    |:---:|:---|:---|
    | 大きい（正） | その方向に動くと損失が急増 | 反対方向（負方向）に大きく移動 |
    | 小さい（ゼロ近く） | どちらに動いても損失がほぼ変わらない | 更新量がほぼゼロ → 停止 |
    | ゼロ | 損失の最小値（極値）に到達 | 更新なし = 学習完了 |

    #### 実用的な収束判定：勾配ノルム

    偏微分が完全にゼロになることは実際にはないため、**勾配ノルム**（勾配ベクトルの大きさ）で判断します：

    $$\|\nabla f\| = \sqrt{\left(\frac{\partial f}{\partial w}\right)^2 + \left(\frac{\partial f}{\partial b}\right)^2} < \varepsilon$$

    閾値 $\varepsilon$（例：$10^{-4}$）を下回ったら「十分に収束した」とみなします。

    #### 手計算で確認（初期値 $w=2,\ b=-2$、学習率 $\alpha=0.2$）

    ```
    step 0:  w= 2.0000, b=-2.0000  →  ∂f/∂w= 2.0000, ∂f/∂b=-2.0000,  ‖∇f‖= 2.828
    step 1:  w= 1.6000, b=-1.6000  →  ∂f/∂w= 1.6000, ∂f/∂b=-1.6000,  ‖∇f‖= 2.263   （20% 減）
    step 2:  w= 1.2800, b=-1.2800  →  ∂f/∂w= 1.2800, ∂f/∂b=-1.2800,  ‖∇f‖= 1.810
    ...
    step10:  w= 0.2147, b=-0.2147  →  ∂f/∂w= 0.2147, ∂f/∂b=-0.2147,  ‖∇f‖= 0.304
    ```

    各ステップで勾配が **0.8 倍**（= 1 − α×2）ずつ減衰していきます。
    """)
    return


@app.cell
def _(gd_b0, gd_lr, gd_steps, gd_w0, np, plt):
    def _f9(w, b): return w**2 + w * b + b**2
    def _gw9(w, b): return 2 * w + b
    def _gb9(w, b): return w + 2 * b

    _lr9 = gd_lr.value
    _n9 = gd_steps.value
    _wi9, _bi9 = gd_w0.value, gd_b0.value

    _s9, _gw9_hist, _gb9_hist, _gnorm9, _w9_hist, _b9_hist, _f9_hist = (
        [0], [_gw9(_wi9, _bi9)], [_gb9(_wi9, _bi9)],
        [np.hypot(_gw9(_wi9, _bi9), _gb9(_wi9, _bi9))],
        [_wi9], [_bi9], [_f9(_wi9, _bi9)],
    )
    for _k in range(_n9):
        _wi9 -= _lr9 * _gw9(_wi9, _bi9)
        _bi9 -= _lr9 * _gb9(_wi9, _bi9)
        _s9.append(_k + 1)
        _gw9_hist.append(_gw9(_wi9, _bi9))
        _gb9_hist.append(_gb9(_wi9, _bi9))
        _gnorm9.append(np.hypot(_gw9(_wi9, _bi9), _gb9(_wi9, _bi9)))
        _w9_hist.append(_wi9)
        _b9_hist.append(_bi9)
        _f9_hist.append(_f9(_wi9, _bi9))

    _eps9 = 0.05
    _conv_idx = next((i for i, v in enumerate(_gnorm9) if v < _eps9), None)

    fig9, axes9 = plt.subplots(2, 2, figsize=(13, 9))

    # --- 左上：∂f/∂w の推移 ---
    _axgw = axes9[0, 0]
    _axgw.axhline(0, color="gray", lw=1.5, linestyle="--", label="ゼロ（収束目標）")
    _axgw.plot(_s9, _gw9_hist, "o-", color="steelblue", lw=2, ms=5, label="∂f/∂w = 2w+b")
    _axgw.fill_between(_s9, _gw9_hist, 0, alpha=0.12, color="steelblue")
    _annot_idx = [0, _n9 // 4, _n9 // 2, 3 * _n9 // 4, _n9] if _n9 >= 4 else list(range(_n9 + 1))
    for _i in dict.fromkeys(_annot_idx):
        _axgw.annotate(f"{_gw9_hist[_i]:.3f}", xy=(_s9[_i], _gw9_hist[_i]),
                       xytext=(0, 8), textcoords="offset points",
                       fontsize=7.5, color="steelblue", ha="center")
    _axgw.set_xlabel("ステップ数")
    _axgw.set_ylabel("∂f/∂w")
    _axgw.set_title("∂f/∂w（w 方向の勾配）の変化\nゼロに近づく = w が最小値に収束中", fontsize=10)
    _axgw.legend(fontsize=8)
    _axgw.grid(True, alpha=0.3)

    # --- 右上：∂f/∂b の推移 ---
    _axgb = axes9[0, 1]
    _axgb.axhline(0, color="gray", lw=1.5, linestyle="--", label="ゼロ（収束目標）")
    _axgb.plot(_s9, _gb9_hist, "o-", color="seagreen", lw=2, ms=5, label="∂f/∂b = w+2b")
    _axgb.fill_between(_s9, _gb9_hist, 0, alpha=0.12, color="seagreen")
    for _i in dict.fromkeys(_annot_idx):
        _axgb.annotate(f"{_gb9_hist[_i]:.3f}", xy=(_s9[_i], _gb9_hist[_i]),
                       xytext=(0, 8), textcoords="offset points",
                       fontsize=7.5, color="seagreen", ha="center")
    _axgb.set_xlabel("ステップ数")
    _axgb.set_ylabel("∂f/∂b")
    _axgb.set_title("∂f/∂b（b 方向の勾配）の変化\nゼロに近づく = b が最小値に収束中", fontsize=10)
    _axgb.legend(fontsize=8)
    _axgb.grid(True, alpha=0.3)

    # --- 左下：勾配ノルムの推移 ---
    _axnorm = axes9[1, 0]
    _axnorm.axhline(0, color="gray", lw=1, alpha=0.4)
    _axnorm.axhline(_eps9, color="crimson", lw=2, linestyle=":", label=f"収束閾値 ε = {_eps9}")
    _axnorm.plot(_s9, _gnorm9, "o-", color="orangered", lw=2, ms=5, label="‖∇f‖ = √(∂w²+∂b²)")
    _axnorm.fill_between(_s9, _gnorm9, 0, alpha=0.12, color="orangered")
    if _conv_idx is not None:
        _axnorm.axvline(_conv_idx, color="purple", lw=2, linestyle="-.",
                        label=f"収束：step {_conv_idx}（‖∇f‖ < {_eps9}）")
        _axnorm.scatter([_s9[_conv_idx]], [_gnorm9[_conv_idx]], color="purple", s=100, zorder=6)
    for _i in dict.fromkeys(_annot_idx):
        _axnorm.annotate(f"{_gnorm9[_i]:.3f}", xy=(_s9[_i], _gnorm9[_i]),
                         xytext=(0, 8), textcoords="offset points",
                         fontsize=7.5, color="orangered", ha="center")
    _axnorm.set_xlabel("ステップ数")
    _axnorm.set_ylabel("勾配ノルム ‖∇f‖")
    _axnorm.set_title(
        f"勾配ノルム：‖∇f‖ < ε={_eps9} になったとき「学習完了」\n"
        + (f"→ step {_conv_idx} で収束" if _conv_idx is not None else f"→ {_n9} ステップでも未収束"),
        fontsize=10)
    _axnorm.legend(fontsize=8)
    _axnorm.grid(True, alpha=0.3)

    # --- 右下：数値テーブル ---
    _axtbl = axes9[1, 1]
    _axtbl.axis("off")
    _show = sorted(set([0] + list(range(0, _n9 + 1, max(1, _n9 // 7))) + [_n9]))
    _rows = [[str(_i),
              f"{_w9_hist[_i]:.4f}", f"{_b9_hist[_i]:.4f}",
              f"{_gw9_hist[_i]:.4f}", f"{_gb9_hist[_i]:.4f}",
              f"{_gnorm9[_i]:.4f}", f"{_f9_hist[_i]:.4f}"]
             for _i in _show]
    _tbl = _axtbl.table(
        cellText=_rows,
        colLabels=["step", "w", "b", "∂f/∂w", "∂f/∂b", "‖∇f‖", "損失 f"],
        loc="center", cellLoc="center")
    _tbl.auto_set_font_size(False)
    _tbl.set_fontsize(8)
    _tbl.scale(1.05, 1.6)
    for (_r, _c), _cell in _tbl.get_celld().items():
        if _r == 0:
            _cell.set_facecolor("#2c5f8a")
            _cell.set_text_props(color="white", fontweight="bold")
        elif _conv_idx is not None and _show[_r - 1] == _conv_idx:
            _cell.set_facecolor("#e8d5f5")
        elif _r % 2 == 0:
            _cell.set_facecolor("#f0f4f8")
    _axtbl.set_title("各ステップの詳細数値（紫行 = 収束したステップ）", fontsize=10, pad=10)

    _title_msg = (f"step {_conv_idx} で収束（‖∇f‖ = {_gnorm9[_conv_idx]:.4f} < ε={_eps9}）"
                  if _conv_idx is not None
                  else f"{_n9} ステップ後も未収束（‖∇f‖ = {_gnorm9[-1]:.4f} > ε={_eps9}）")
    plt.suptitle(f"偏微分の変化と収束の判定：{_title_msg}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig9
    return axes9, fig9


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **グラフの読み方**
    >
    > - **∂f/∂w・∂f/∂b グラフ**：両方ともゼロ点（破線）に向かって減衰します。
    >   これが「w と b が損失の最小点 (0, 0) へ近づいている」証拠です。
    >
    > - **勾配ノルム ‖∇f‖ グラフ**：2 つの偏微分を合成した「全体的な勾配の大きさ」。
    >   赤い点線（閾値 ε）を下回った瞬間（紫の縦線）が **収束＝学習完了** の判定点です。
    >
    > - **数値テーブル**：紫色の行が収束した瞬間のステップです。

    #### スライダーで確認できること

    | 操作 | 観察できること |
    |:---|:---|
    | 学習率 α を大きくする（0.4〜0.5） | 勾配が振動・発散し、収束が遅れる or 失敗する |
    | 学習率 α を小さくする（0.05） | 勾配がゆっくり減衰、多くのステップが必要 |
    | ステップ数を増やす | ‖∇f‖ が十分ゼロに近づくまで追跡できる |
    | 初期値を原点近くに設定 | 最初から勾配が小さく、少ないステップで収束 |

    #### PyTorch の学習ループとの対応

    ```python
    for epoch in range(max_epochs):
        optimizer.zero_grad()   # 勾配リセット
        loss = model(x)         # 順伝播（損失計算）
        loss.backward()         # 逆伝播（偏微分を自動計算）
        optimizer.step()        # パラメータ更新（w ← w - α * ∂L/∂w）

        grad_norm = sum(p.grad.norm()**2 for p in model.parameters())**0.5
        if grad_norm < 1e-4:    # ‖∇L‖ < ε → 収束判定
            print(f"Converged at epoch {epoch}")
            break
    ```
    """)
    return


# ============================================================
# まとめ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## まとめ

    | コード | 数学記法 | 演算の種類 |
    |--------|---------|-----------|
    | `tensor @ tensor.T` | $A A^\\top$ | 行列積 |
    | `tensor * tensor` | $A \\odot A$ | 要素積（アダマール積） |
    | `tensor.pow(n)` | $A^{\\odot n}$ | 要素ごとのべき乗 |
    | `tensor[i]` | $\\mathbf{a}_{i,:}$ | 第 $i$ 行ベクトルの抽出 |
    | `tensor[:, j]` | $\\mathbf{a}_{:,j}$ | 第 $j$ 列ベクトルの抽出 |
    | `tensor.T` | $A^\\top$ | 転置 |
    | `loss.backward()` | $\\nabla L$ を計算 | 連鎖律による全偏微分の自動計算 |
    | `w.grad` | $\\frac{\\partial L}{\\partial w}$ | w に関する損失の偏微分 |
    """)
    return


if __name__ == "__main__":
    app.run()
