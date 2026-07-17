import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # 行列の内積と外積：何を表現しているのか？

    行列の **内積（inner product）** と **外積（outer product）** は、線形代数の中でも特に重要な2つの演算です。
    それぞれが「何を計算しているのか」だけでなく、「どんな現象・関係を表現しているのか」を視覚的に理解しましょう。
    """)
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📐 内積（Inner Product / Dot Product）

    ### 計算方法
    2つのベクトル $\\mathbf{a} = [a_1, a_2, ..., a_n]$ と $\\mathbf{b} = [b_1, b_2, ..., b_n]$ の内積：

    $$\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^{n} a_i b_i = |\\mathbf{a}| |\\mathbf{b}| \\cos\\theta$$

    ### 何を表現しているか？
    > **「ベクトルaをベクトルbの方向に射影したときの大きさ × bの大きさ」**
    > つまり、**2つのベクトルがどれだけ同じ方向を向いているか（類似度）** を表す。

    - $\\cos\\theta = 1$ → 完全に同じ方向（内積が最大）
    - $\\cos\\theta = 0$ → 直交（無関係）
    - $\\cos\\theta = -1$ → 完全に逆方向（内積が最小）
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torch

    # macOS Japanese font setup
    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    return np, plt, patches, torch


@app.cell
def _(mo):
    # インタラクティブスライダー：ベクトルbの角度を変える
    angle_slider = mo.ui.slider(
        start=0,
        stop=360,
        step=5,
        value=45,
        label="ベクトル b の角度 (度)",
        full_width=True,
    )
    angle_slider
    return (angle_slider,)


@app.cell
def _(angle_slider, np, plt, torch):
    # ベクトルaは固定（右向き）
    theta_deg = angle_slider.value
    theta_rad = np.radians(theta_deg)

    a = torch.tensor([1.5, 0.0], dtype=torch.float32)
    b = torch.tensor([np.cos(theta_rad) * 1.5, np.sin(theta_rad) * 1.5], dtype=torch.float32)

    dot_product = torch.dot(a, b).item()
    cos_theta = dot_product / (torch.norm(a).item() * torch.norm(b).item())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左図：ベクトルと射影の可視化 ---
    ax1 = axes[0]
    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-2.2, 2.2)
    ax1.set_aspect("equal")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # ベクトルa（青）
    ax1.annotate(
        "",
        xy=(a[0].item(), a[1].item()),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=2.5),
    )
    ax1.text(a[0].item() + 0.1, a[1].item() + 0.1, "a", fontsize=14, color="steelblue", fontweight="bold")

    # ベクトルb（オレンジ）
    ax1.annotate(
        "",
        xy=(b[0].item(), b[1].item()),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="orangered", lw=2.5),
    )
    ax1.text(b[0].item() + 0.1, b[1].item() + 0.1, "b", fontsize=14, color="orangered", fontweight="bold")

    # aへのbの射影（緑の点線）
    proj_len = dot_product / torch.norm(a).item()
    a_unit = a / torch.norm(a)
    proj_point = proj_len * a_unit

    ax1.plot(
        [b[0].item(), proj_point[0].item()],
        [b[1].item(), proj_point[1].item()],
        "g--",
        alpha=0.7,
        linewidth=1.5,
        label="射影の垂線",
    )
    ax1.scatter([proj_point[0].item()], [proj_point[1].item()], color="green", s=80, zorder=5)

    # 射影ベクトル（緑矢印）
    if abs(proj_len) > 0.05:
        ax1.annotate(
            "",
            xy=(proj_point[0].item(), proj_point[1].item()),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
        )
    ax1.text(
        proj_point[0].item() / 2,
        -0.3,
        f"射影 = {proj_len:.2f}",
        fontsize=10,
        color="green",
        ha="center",
    )

    # 角度の弧
    arc_angles = np.linspace(0, theta_rad, 50)
    arc_r = 0.4
    ax1.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), "purple", lw=1.5)
    mid_angle = theta_rad / 2
    ax1.text(
        0.55 * np.cos(mid_angle),
        0.55 * np.sin(mid_angle),
        f"θ={theta_deg}°",
        fontsize=10,
        color="purple",
    )

    ax1.set_title(
        f"内積 a·b = {dot_product:.3f}\n(= |a||b|cosθ = 1.5 × 1.5 × {cos_theta:.3f})",
        fontsize=11,
        pad=10,
    )
    ax1.legend(loc="lower right", fontsize=9)

    # --- 右図：角度と内積の関係グラフ ---
    ax2 = axes[1]
    angles = np.linspace(0, 2 * np.pi, 360)
    dot_values = 1.5 * 1.5 * np.cos(angles)

    ax2.plot(np.degrees(angles), dot_values, color="steelblue", lw=2)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axvline(theta_deg, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label=f"現在: θ={theta_deg}°")
    ax2.scatter([theta_deg], [dot_product], color="red", s=100, zorder=5)
    ax2.fill_between(np.degrees(angles), dot_values, 0,
                     where=(dot_values > 0), alpha=0.15, color="blue", label="正の内積（同方向寄り）")
    ax2.fill_between(np.degrees(angles), dot_values, 0,
                     where=(dot_values < 0), alpha=0.15, color="red", label="負の内積（逆方向寄り）")

    ax2.set_xlabel("角度 θ (度)", fontsize=11)
    ax2.set_ylabel("内積 a·b", fontsize=11)
    ax2.set_title("角度と内積の関係：cosθ カーブ", fontsize=11)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("内積のジオメトリ：射影と類似度", fontsize=13, fontweight="bold", y=1.01)
    fig
    return (
        a,
        a_unit,
        angles,
        arc_angles,
        arc_r,
        ax1,
        ax2,
        b,
        cos_theta,
        dot_product,
        dot_values,
        fig,
        mid_angle,
        proj_len,
        proj_point,
        theta_deg,
        theta_rad,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **スライダーで角度を変えて観察してみよう！**
    > - **θ = 0°**：同じ方向 → 内積が最大（2.25）
    > - **θ = 90°**：直交 → 内積 = 0（無関係）
    > - **θ = 180°**：逆方向 → 内積が最小（−2.25）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🔲 外積（Outer Product）

    ### 計算方法
    列ベクトル $\\mathbf{a} \\in \\mathbb{R}^m$ と行ベクトル $\\mathbf{b} \\in \\mathbb{R}^n$ の外積：

    $$\\mathbf{a} \\otimes \\mathbf{b} = \\mathbf{a} \\mathbf{b}^T = \\begin{bmatrix} a_1 b_1 & a_1 b_2 & \\cdots \\\\ a_2 b_1 & a_2 b_2 & \\cdots \\\\ \\vdots & & \\ddots \\end{bmatrix}$$

    ### 何を表現しているか？
    > **「aの各要素がbの各要素に与える影響の強さ」を全組み合わせで記録した行列**
    > = **ランク1行列** （すべての行が $\\mathbf{b}$ の定数倍、すべての列が $\\mathbf{a}$ の定数倍）

    **直感的なイメージ**：
    - aが「行ごとの重要度スコア」、bが「列ごとの重要度スコア」だとすると
    - 外積は「各マス目の重要度 = 行スコア × 列スコア」という**影響マップ**
    """)
    return


@app.cell
def _(mo):
    # ベクトルaの入力スライダー
    mo.md("### インタラクティブ外積デモ：ベクトルの値を変えてみよう")
    return


@app.cell
def _(mo):
    a_sliders = mo.ui.array([
        mo.ui.slider(start=-3.0, stop=3.0, step=0.5, value=v, label=f"a[{i}]")
        for i, v in enumerate([2.0, 1.0, -1.0, 0.5])
    ])
    b_sliders = mo.ui.array([
        mo.ui.slider(start=-3.0, stop=3.0, step=0.5, value=v, label=f"b[{i}]")
        for i, v in enumerate([1.0, 2.0, -0.5, 1.5])
    ])
    mo.hstack([
        mo.vstack([mo.md("**ベクトル a（行方向の重み）**"), *a_sliders]),
        mo.vstack([mo.md("**ベクトル b（列方向の重み）**"), *b_sliders]),
    ])
    return a_sliders, b_sliders


@app.cell
def _(a_sliders, b_sliders, np, plt, torch):
    a_vec = torch.tensor([s.value for s in a_sliders], dtype=torch.float32)
    b_vec = torch.tensor([s.value for s in b_sliders], dtype=torch.float32)

    # 外積の計算
    outer = torch.outer(a_vec, b_vec)

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    # --- ベクトルa の棒グラフ ---
    ax_a = axes2[0]
    colors_a = ["steelblue" if v >= 0 else "orangered" for v in a_vec.tolist()]
    bars_a = ax_a.barh(range(len(a_vec)), a_vec.tolist(), color=colors_a, edgecolor="white")
    ax_a.set_yticks(range(len(a_vec)))
    ax_a.set_yticklabels([f"a[{i}]" for i in range(len(a_vec))])
    ax_a.axvline(0, color="black", linewidth=0.8)
    ax_a.set_xlim(-3.5, 3.5)
    ax_a.set_title("ベクトル a\n（行方向の重み）", fontsize=11)
    ax_a.invert_yaxis()  # a[0]をヒートマップのrow 0（上端）に揃える
    ax_a.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars_a, a_vec.tolist()):
        ax_a.text(val + (0.15 if val >= 0 else -0.15), bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

    # --- 外積ヒートマップ ---
    ax_mid = axes2[1]
    outer_np = outer.numpy()
    vmax = max(abs(outer_np.max()), abs(outer_np.min()), 0.1)
    im = ax_mid.imshow(outer_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    fig2.colorbar(im, ax=ax_mid, shrink=0.8)
    ax_mid.set_xticks(range(len(b_vec)))
    ax_mid.set_xticklabels([f"b[{i}]" for i in range(len(b_vec))])
    ax_mid.set_yticks(range(len(a_vec)))
    ax_mid.set_yticklabels([f"a[{i}]" for i in range(len(a_vec))])
    ax_mid.set_title(f"外積 a ⊗ b（影響マップ）\nランク = {int(np.linalg.matrix_rank(outer_np))}", fontsize=11)
    # セルの値を表示
    for i in range(len(a_vec)):
        for j in range(len(b_vec)):
            ax_mid.text(j, i, f"{outer_np[i,j]:.1f}", ha="center", va="center",
                       fontsize=8, color="black" if abs(outer_np[i,j]) < vmax * 0.6 else "white")

    # --- ベクトルb の棒グラフ ---
    ax_b = axes2[2]
    colors_b = ["steelblue" if v >= 0 else "orangered" for v in b_vec.tolist()]
    bars_b = ax_b.bar(range(len(b_vec)), b_vec.tolist(), color=colors_b, edgecolor="white")
    ax_b.set_xticks(range(len(b_vec)))
    ax_b.set_xticklabels([f"b[{i}]" for i in range(len(b_vec))])
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.set_ylim(-3.5, 3.5)
    ax_b.set_title("ベクトル b\n（列方向の重み）", fontsize=11)
    ax_b.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars_b, b_vec.tolist()):
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + (0.15 if val >= 0 else -0.25),
                  f"{val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.suptitle("外積：行×列の全組み合わせ影響マップ", fontsize=13, fontweight="bold", y=1.02)
    fig2
    return (
        a_vec,
        ax_a,
        ax_b,
        ax_mid,
        axes2,
        b_vec,
        bars_a,
        bars_b,
        colors_a,
        colors_b,
        fig2,
        im,
        outer,
        outer_np,
        vmax,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🧩 ランク（Rank）とは何か？

    外積が「常にランク1」という性質を理解するために、まず**行列のランク**を説明します。

    ### 定義と計算方法

    $$\\text{rank}(A) = \\text{行列 } A \\text{ の中で「互いに独立な行（または列）の数」}$$

    PyTorch では：
    ```python
    torch.linalg.matrix_rank(A)
    ```

    ### 何を表現しているか？

    > **「その行列が表現できる情報の次元数」**

    | ランク | 意味 | 例 |
    |:---:|:---|:---|
    | 1 | 1方向の情報しかない（全行が互いに定数倍） | 外積 $\\mathbf{a} \\otimes \\mathbf{b}$ |
    | 2 | 2方向の独立な情報がある | 2つの外積の和 |
    | r | r 個の独立な「方向」で全体が決まる | SVDの第r成分まで |

    ### なぜ外積は必ずランク1なのか？

    外積 $\\mathbf{a} \\otimes \\mathbf{b}$ の第 $i$ 行は $a_i \\cdot \\mathbf{b}$ です。
    **どの行も $\\mathbf{b}$ の定数倍**なので、独立な行は1本だけ → ランク = 1
    """)
    return


@app.cell
def _(np, plt, torch):
    _rng_rank = np.random.default_rng(0)

    _av_r = torch.tensor([3.0, 1.5, 0.5, 2.0])   # 全正値：全行が同じ向きに揃う
    _bv_r = torch.tensor([1.0, 2.0, -1.0, 1.5])
    rank1_mat = torch.outer(_av_r, _bv_r).numpy()

    _av2_r = torch.tensor([0.0, 1.0, 0.0, -1.0])
    _bv2_r = torch.tensor([1.0, 0.0, -1.0, 0.0])
    rank2_mat = rank1_mat + torch.outer(_av2_r, _bv2_r).numpy() * 2.5

    full_mat = _rng_rank.standard_normal((4, 4))

    fig_rank, _axes_rank = plt.subplots(2, 3, figsize=(14, 7))

    _row_colors = ["steelblue", "orangered", "green", "purple"]

    for _col_r, (_mat_r, _title_r) in enumerate([
        (rank1_mat, "ランク1行列\n（外積 a⊗b）"),
        (rank2_mat, "ランク2行列\n（2つの外積の和）"),
        (full_mat, "ランク4行列\n（フルランク）"),
    ]):
        _rank_val = int(np.linalg.matrix_rank(_mat_r))
        _ax_heat_r = _axes_rank[0, _col_r]
        _vmax_r = max(abs(_mat_r.max()), abs(_mat_r.min()), 0.1)
        _im_r = _ax_heat_r.imshow(_mat_r, cmap="RdBu_r", vmin=-_vmax_r, vmax=_vmax_r)
        plt.colorbar(_im_r, ax=_ax_heat_r, shrink=0.7)
        for _ri in range(4):
            for _rj in range(4):
                _ax_heat_r.text(_rj, _ri, f"{_mat_r[_ri,_rj]:.1f}", ha="center", va="center",
                                fontsize=8, color="white" if abs(_mat_r[_ri,_rj]) > _vmax_r * 0.6 else "black")
        _ax_heat_r.set_title(_title_r + f"\nrank = {_rank_val}", fontsize=10)

        _ax_rows_r = _axes_rank[1, _col_r]
        for _ri, _row in enumerate(_mat_r):
            _norm_val = np.linalg.norm(_row)
            _row_n = _row / _norm_val if _norm_val > 0.01 else _row
            _ax_rows_r.plot(range(4), _row_n, marker="o", label=f"行{_ri}（正規化）",
                            color=_row_colors[_ri], lw=2, markersize=6)
        _ax_rows_r.set_title("各行を正規化したグラフ\nランク1なら全行が完全に重なる", fontsize=9)
        _ax_rows_r.set_xlabel("列インデックス")
        _ax_rows_r.set_ylabel("正規化後の値")
        _ax_rows_r.legend(fontsize=8, loc="upper right")
        _ax_rows_r.grid(True, alpha=0.3)
        _ax_rows_r.set_xticks(range(4))
        _ax_rows_r.set_ylim(-1.6, 1.6)

    plt.tight_layout()
    plt.suptitle("ランクの比較：正規化グラフが重なるほど低ランク", fontsize=13, fontweight="bold", y=1.01)
    fig_rank
    return fig_rank, full_mat, rank1_mat, rank2_mat


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > 正規化グラフの見方：
    > - **ランク1**：4本の線が完全に重なる（すべての行が同じ「向き」）
    > - **ランク2**：2グループに分かれる
    > - **ランク4**：4本の線がバラバラ（互いに独立）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## ➕ 2つの外積の和：「パターンの重ね合わせ」

    ランク2行列を「2つの外積の和」と表現しましたが、これは具体的に何を意味するのでしょうか？

    ### 具体的な計算例

    **1つ目の外積**（横縞パターン）

    $$\\mathbf{a}_1 = \\begin{bmatrix}3\\\\1\\\\2\\end{bmatrix}, \\quad \\mathbf{b}_1 = \\begin{bmatrix}1\\\\2\\\\-1\\end{bmatrix} \\implies \\mathbf{a}_1 \\otimes \\mathbf{b}_1 = \\begin{bmatrix}3&6&-3\\\\1&2&-1\\\\2&4&-2\\end{bmatrix} \\quad \\text{（ランク1）}$$

    全行が $\\mathbf{b}_1 = [1, 2, -1]$ の定数倍 → 「1方向」の情報だけ。

    **2つ目の外積**（別のパターン）

    $$\\mathbf{a}_2 = \\begin{bmatrix}0\\\\1\\\\-1\\end{bmatrix}, \\quad \\mathbf{b}_2 = \\begin{bmatrix}2\\\\0\\\\1\\end{bmatrix} \\implies \\mathbf{a}_2 \\otimes \\mathbf{b}_2 = \\begin{bmatrix}0&0&0\\\\2&0&1\\\\-2&0&-1\\end{bmatrix} \\quad \\text{（ランク1）}$$

    **2つを足す**

    $$\\mathbf{a}_1 \\otimes \\mathbf{b}_1 + \\mathbf{a}_2 \\otimes \\mathbf{b}_2 = \\begin{bmatrix}3&6&-3\\\\3&2&0\\\\0&4&-3\\end{bmatrix} \\quad \\text{（ランク2）}$$

    行が互いに定数倍ではなくなり、**2方向の独立した情報**を持つランク2行列になります。

    > **直感**：外積1つ＝1種類の「塗り方」。足し合わせるほど複雑な模様（高いランク）になる。
    """)
    return


@app.cell
def _(np, plt, torch):
    _a1 = torch.tensor([3.0, 1.0, 2.0])
    _b1 = torch.tensor([1.0, 2.0, -1.0])
    _a2 = torch.tensor([0.0, 1.0, -1.0])
    _b2 = torch.tensor([2.0, 0.0, 1.0])
    _op1 = torch.outer(_a1, _b1).numpy()
    _op2 = torch.outer(_a2, _b2).numpy()
    _opsum = _op1 + _op2
    _vmax_op = max(abs(_op1).max(), abs(_op2).max(), abs(_opsum).max())

    fig_sum, _axes_sum = plt.subplots(1, 5, figsize=(16, 3.5),
                                      gridspec_kw={"width_ratios": [3, 0.4, 3, 0.4, 3]})

    # --- 外積1 ---
    _im1 = _axes_sum[0].imshow(_op1, cmap="RdBu_r", vmin=-_vmax_op, vmax=_vmax_op)
    fig_sum.colorbar(_im1, ax=_axes_sum[0], shrink=0.7)
    for _ri in range(3):
        for _ci in range(3):
            _axes_sum[0].text(_ci, _ri, f"{_op1[_ri,_ci]:.0f}", ha="center", va="center",
                              fontsize=11, fontweight="bold",
                              color="white" if abs(_op1[_ri,_ci]) > _vmax_op*0.55 else "black")
    _axes_sum[0].set_title(f"a₁⊗b₁  =[3,1,2]ᵀ×[1,2,-1]\nrank={int(np.linalg.matrix_rank(_op1))}", fontsize=10)
    _axes_sum[0].set_xticks(range(3))
    _axes_sum[0].set_yticks(range(3))
    _axes_sum[0].set_xticklabels(["b₁[0]","b₁[1]","b₁[2]"])
    _axes_sum[0].set_yticklabels(["a₁[0]","a₁[1]","a₁[2]"])

    # --- + 記号 ---
    _axes_sum[1].text(0.5, 0.5, "+", ha="center", va="center", fontsize=28,
                      fontweight="bold", color="gray", transform=_axes_sum[1].transAxes)
    _axes_sum[1].axis("off")

    # --- 外積2 ---
    _im2 = _axes_sum[2].imshow(_op2, cmap="RdBu_r", vmin=-_vmax_op, vmax=_vmax_op)
    fig_sum.colorbar(_im2, ax=_axes_sum[2], shrink=0.7)
    for _ri in range(3):
        for _ci in range(3):
            _axes_sum[2].text(_ci, _ri, f"{_op2[_ri,_ci]:.0f}", ha="center", va="center",
                              fontsize=11, fontweight="bold",
                              color="white" if abs(_op2[_ri,_ci]) > _vmax_op*0.55 else "black")
    _axes_sum[2].set_title(f"a₂⊗b₂  =[0,1,-1]ᵀ×[2,0,1]\nrank={int(np.linalg.matrix_rank(_op2))}", fontsize=10)
    _axes_sum[2].set_xticks(range(3))
    _axes_sum[2].set_yticks(range(3))
    _axes_sum[2].set_xticklabels(["b₂[0]","b₂[1]","b₂[2]"])
    _axes_sum[2].set_yticklabels(["a₂[0]","a₂[1]","a₂[2]"])

    # --- = 記号 ---
    _axes_sum[3].text(0.5, 0.5, "=", ha="center", va="center", fontsize=28,
                      fontweight="bold", color="gray", transform=_axes_sum[3].transAxes)
    _axes_sum[3].axis("off")

    # --- 和 ---
    _im3 = _axes_sum[4].imshow(_opsum, cmap="RdBu_r", vmin=-_vmax_op, vmax=_vmax_op)
    fig_sum.colorbar(_im3, ax=_axes_sum[4], shrink=0.7)
    for _ri in range(3):
        for _ci in range(3):
            _axes_sum[4].text(_ci, _ri, f"{_opsum[_ri,_ci]:.0f}", ha="center", va="center",
                              fontsize=11, fontweight="bold",
                              color="white" if abs(_opsum[_ri,_ci]) > _vmax_op*0.55 else "black")
    _axes_sum[4].set_title(f"a₁⊗b₁ + a₂⊗b₂\n（2つのパターンの重ね合わせ）  rank={int(np.linalg.matrix_rank(_opsum))}", fontsize=10)
    _axes_sum[4].set_xticks(range(3))
    _axes_sum[4].set_yticks(range(3))
    _axes_sum[4].set_xticklabels(["[0]","[1]","[2]"])
    _axes_sum[4].set_yticklabels(["[0]","[1]","[2]"])

    plt.suptitle("2つの外積の和：ランク1 + ランク1 = ランク2", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_sum
    return (fig_sum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **観察ポイント**
    > - 左の行列（a₁⊗b₁）は全行が $[1,2,-1]$ の定数倍 → **1方向**
    > - 中の行列（a₂⊗b₂）は全行が $[2,0,1]$ の定数倍 → **別の1方向**
    > - 右の和は「どちらの向きでも全行が定数倍」にはならない → **2方向 = ランク2**
    >
    > これが「外積の和を重ねるほどランクが上がる」という意味です。
    > SVDはその逆：複雑な行列を「外積の和」に**分解**する操作です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🔬 SVD（特異値分解）とは何か？

    ### 計算方法

    任意の行列 $A \\in \\mathbb{R}^{m \\times n}$ は必ず次のように分解できます：

    $$A = U \\Sigma V^T = \\sum_{i=1}^{r} \\sigma_i \\, \\mathbf{u}_i \\otimes \\mathbf{v}_i$$

    | 記号 | 形状 | 意味 |
    |:---:|:---:|:---|
    | $U$ | $m \\times m$ | 行空間の基底（左特異ベクトル）|
    | $\\Sigma$ | $m \\times n$ | 対角行列（特異値 $\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq 0$）|
    | $V^T$ | $n \\times n$ | 列空間の基底（右特異ベクトル）|

    PyTorch では：
    ```python
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # S が特異値のベクトル（降順）
    # 第k成分の寄与 = S[k] * torch.outer(U[:, k], Vh[k, :])
    ```

    ### 何を表現しているか？

    > **「行列を、重要度（特異値）の高い順に並んだ外積の和として分解する」**

    - $\\sigma_1$ が最大 → 第1成分が行列の「最も支配的なパターン」
    - 上位 $k$ 個の外積だけで元の行列を **近似** できる（低ランク近似）
    - $k$ が小さいほど圧縮率が高く、$k = r$ で完全に元の行列が復元される

    ### 使いどころ

    | 場面 | 意味 |
    |:---|:---|
    | 画像圧縮 | 上位k成分だけ保存 → ファイルサイズ削減 |
    | ノイズ除去 | 小さい特異値（ノイズ成分）を捨てる |
    | 推薦システム | ユーザー×アイテム行列を分解して潜在因子を抽出 |
    | 主成分分析（PCA） | データ行列のSVD = 主成分の発見 |
    """)
    return


@app.cell
def _(mo):
    svd_k_slider = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=1,
        label="使う成分数 k（外積の数）",
        full_width=True,
    )
    svd_k_slider
    return (svd_k_slider,)


@app.cell
def _(np, plt, svd_k_slider, torch):
    # 8x8の「2パターンが混ざった」行列を作成
    x8 = np.linspace(0, 1, 8)
    pattern_h = np.sin(2 * np.pi * x8)
    pattern_v = np.cos(2 * np.pi * x8)

    svd_mat = (
        3.0 * np.outer(pattern_v, pattern_h)
        + 1.5 * np.outer(np.sin(4 * np.pi * x8), np.cos(4 * np.pi * x8))
        + 0.5 * np.random.default_rng(7).standard_normal((8, 8))
    )
    A_svd = torch.tensor(svd_mat, dtype=torch.float32)

    U_svd, S_svd, Vh_svd = torch.linalg.svd(A_svd, full_matrices=False)

    k = svd_k_slider.value

    # k個の外積で近似
    A_approx = sum(
        S_svd[i].item() * torch.outer(U_svd[:, i], Vh_svd[i, :])
        for i in range(k)
    )
    A_approx_np = A_approx.numpy()
    error = torch.norm(A_svd - A_approx).item()
    total_energy = torch.norm(A_svd).item()
    energy_ratio = (1 - (error / total_energy) ** 2) * 100

    fig_svd, axes_svd = plt.subplots(1, 4, figsize=(16, 4))

    vmax_svd = max(abs(svd_mat.max()), abs(svd_mat.min()))

    # 元の行列
    ax_orig = axes_svd[0]
    im_orig = ax_orig.imshow(svd_mat, cmap="RdBu_r", vmin=-vmax_svd, vmax=vmax_svd)
    plt.colorbar(im_orig, ax=ax_orig, shrink=0.8)
    ax_orig.set_title("元の行列 A\n（8×8）", fontsize=10)

    # k成分での近似
    ax_approx = axes_svd[1]
    im_approx = ax_approx.imshow(A_approx_np, cmap="RdBu_r", vmin=-vmax_svd, vmax=vmax_svd)
    plt.colorbar(im_approx, ax=ax_approx, shrink=0.8)
    ax_approx.set_title(f"k={k}成分での近似\n情報保持率 {energy_ratio:.1f}%", fontsize=10)

    # 第k成分（最後に追加した外積）
    k_comp = S_svd[k - 1].item() * torch.outer(U_svd[:, k - 1], Vh_svd[k - 1, :])
    k_comp_np = k_comp.numpy()
    vmax_kc = max(abs(k_comp_np.max()), abs(k_comp_np.min()), 0.01)
    ax_kcomp = axes_svd[2]
    im_kc = ax_kcomp.imshow(k_comp_np, cmap="RdBu_r", vmin=-vmax_kc, vmax=vmax_kc)
    plt.colorbar(im_kc, ax=ax_kcomp, shrink=0.8)
    ax_kcomp.set_title(
        f"第{k}成分（ランク1外積）\nσ_{k} = {S_svd[k-1].item():.2f}", fontsize=10
    )

    # 特異値の棒グラフ
    ax_sigma = axes_svd[3]
    sigma_vals = S_svd.numpy()
    bar_colors_svd = ["orangered" if i < k else "lightgray" for i in range(len(sigma_vals))]
    ax_sigma.bar(range(1, len(sigma_vals) + 1), sigma_vals, color=bar_colors_svd, edgecolor="white")
    ax_sigma.set_xlabel("成分番号 i")
    ax_sigma.set_ylabel("特異値 σᵢ")
    ax_sigma.set_title("特異値の大きさ\n（赤＝使用中の成分）", fontsize=10)
    ax_sigma.set_xticks(range(1, len(sigma_vals) + 1))
    ax_sigma.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.suptitle(
        f"SVD：外積の和として行列を分解する  A = Σ σᵢ uᵢ⊗vᵢ  （k={k}/{len(sigma_vals)}成分使用）",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig_svd
    return (
        A_approx,
        A_approx_np,
        A_svd,
        S_svd,
        U_svd,
        Vh_svd,
        ax_approx,
        ax_kcomp,
        ax_orig,
        ax_sigma,
        axes_svd,
        bar_colors_svd,
        energy_ratio,
        error,
        fig_svd,
        im_approx,
        im_kc,
        im_orig,
        k,
        k_comp,
        k_comp_np,
        pattern_h,
        pattern_v,
        sigma_vals,
        svd_mat,
        total_energy,
        vmax_kc,
        vmax_svd,
        x8,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **スライダーで k を変えて観察してみよう！**
    > - **k = 1**：最も重要な1パターンだけ → 大まかな形だけ復元
    > - **k = 2〜3**：主要なパターンが揃い、形がはっきりしてくる
    > - **k = 8**：すべての成分を使用 → 完全に元の行列を復元（情報保持率100%）
    >
    > 棒グラフの赤い部分が「今使っている成分（外積）」です。
    > 特異値が大きい成分ほど元の行列への貢献が大きく、小さい成分はノイズに近い。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 🌊 実応用：画像の外積 ＝ テクスチャパターンの生成

    外積の直感的な応用例として、**2つの1次元信号から2次元パターンを生成する** ことを考えます。
    これは画像のセパラブルフィルタや、ニューラルネットワークのアテンション機構の基本原理です。
    """)
    return


@app.cell
def _(mo):
    pattern_select = mo.ui.dropdown(
        options={
            "波 × 波（干渉縞）": "wave_wave",
            "ガウス × ガウス（スポット）": "gauss_gauss",
            "波 × ガウス（波紋）": "wave_gauss",
            "ランダム × ランダム": "rand_rand",
        },
        value="波 × 波（干渉縞）",
        label="パターンの組み合わせを選択",
    )
    freq_slider = mo.ui.slider(start=1, stop=8, step=1, value=3, label="周波数")
    mo.hstack([pattern_select, freq_slider])
    return freq_slider, pattern_select


@app.cell
def _(freq_slider, np, pattern_select, plt, torch):
    n_pts = 64
    x = np.linspace(0, 2 * np.pi, n_pts)
    freq = freq_slider.value

    pattern_key = pattern_select.value

    def make_signal(kind, x, freq):
        if kind == "wave":
            return np.sin(freq * x)
        elif kind == "gauss":
            return np.exp(-((x - np.pi) ** 2) / (2 * (np.pi / freq) ** 2))
        elif kind == "rand":
            rng = np.random.default_rng(42)
            return rng.standard_normal(len(x)) * 0.5
        return np.ones(len(x))

    kind_map = {
        "wave_wave": ("wave", "wave"),
        "gauss_gauss": ("gauss", "gauss"),
        "wave_gauss": ("wave", "gauss"),
        "rand_rand": ("rand", "rand"),
    }
    kind_a, kind_b = kind_map[pattern_key]
    sig_a = make_signal(kind_a, x, freq)
    sig_b = make_signal(kind_b, x, freq)

    ta = torch.tensor(sig_a, dtype=torch.float32)
    tb = torch.tensor(sig_b, dtype=torch.float32)
    pattern_2d = torch.outer(ta, tb).numpy()

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5),
                                gridspec_kw={"width_ratios": [1, 1.5, 1]})

    ax3_a = axes3[0]
    ax3_a.plot(sig_a, x, color="steelblue", lw=2)
    ax3_a.fill_betweenx(x, 0, sig_a, alpha=0.3, color="steelblue")
    ax3_a.axvline(0, color="black", lw=0.8)
    ax3_a.set_title(f"信号 a（{kind_a}）\n縦方向パターン", fontsize=10)
    ax3_a.set_xlabel("振幅")
    ax3_a.invert_xaxis()
    ax3_a.grid(True, alpha=0.3)

    ax3_mid = axes3[1]
    vmax3 = max(abs(pattern_2d.max()), abs(pattern_2d.min()), 0.01)
    im3 = ax3_mid.imshow(pattern_2d, cmap="RdBu_r", vmin=-vmax3, vmax=vmax3,
                          origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    plt.colorbar(im3, ax=ax3_mid, shrink=0.8)
    ax3_mid.set_title(f"外積 a ⊗ b\n2Dパターン（{pattern_key}）", fontsize=10)

    ax3_b = axes3[2]
    ax3_b.plot(x, sig_b, color="orangered", lw=2)
    ax3_b.fill_between(x, 0, sig_b, alpha=0.3, color="orangered")
    ax3_b.axhline(0, color="black", lw=0.8)
    ax3_b.set_title(f"信号 b（{kind_b}）\n横方向パターン", fontsize=10)
    ax3_b.set_ylabel("振幅")
    ax3_b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("外積による2Dパターン生成：縦信号 × 横信号 = 2D画像", fontsize=12, fontweight="bold", y=1.02)
    fig3
    return (
        ax3_a,
        ax3_b,
        ax3_mid,
        axes3,
        fig3,
        freq,
        im3,
        kind_a,
        kind_b,
        kind_map,
        make_signal,
        n_pts,
        pattern_2d,
        pattern_key,
        sig_a,
        sig_b,
        ta,
        tb,
        vmax3,
        x,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 📊 内積 vs 外積：まとめ

    | 性質 | 内積 | 外積 |
    |:---:|:---:|:---:|
    | **入力** | ベクトル a (n次元), b (n次元) | ベクトル a (m次元), b (n次元) |
    | **出力** | スカラー | m×n 行列 |
    | **PyTorch** | `torch.dot(a, b)` | `torch.outer(a, b)` |
    | **表現すること** | 2ベクトルの類似度・射影 | 全組み合わせの影響マップ |
    | **結果の特徴** | 角度に依存するcos値 | 必ずランク1の行列 |
    | **応用例** | 類似度検索・コサイン類似度・注意重み | SVD・アテンション行列・セパラブルフィルタ |

    ### 深い関係：内積と外積の双対性
    - **内積** = 2つのベクトルを **1つのスカラー** に「圧縮」する（情報を失う）
    - **外積** = 2つのベクトルを **行列** に「展開」する（全情報を保持）
    - SVD（特異値分解）では：$A = \\sum_i \\sigma_i \\mathbf{u}_i \\otimes \\mathbf{v}_i$（外積の和として行列を分解）
    """)
    return


if __name__ == "__main__":
    app.run()
