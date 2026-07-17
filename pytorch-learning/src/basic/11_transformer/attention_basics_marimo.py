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
    # Attentionの仕組み：Transformerの心臓部を分解する

    ChatGPTをはじめとする現代のAIはすべて **Transformer** というアーキテクチャの上に
    構築されています。その心臓部が **Self-Attention（自己注意機構）** です。

    アイデアは一言でいえば：

    > **各単語が、文中の他の単語を「見て」、自分の意味表現を文脈に合わせて更新する**

    例：「銀行の**口座**」と「川の**口座**」——「口座」は誤りで、正しくは「土手（bank）」の
    ように、**同じ単語でも周囲の単語によって意味が変わる**。Attentionはこの
    「どの単語をどれだけ参照するか」を学習可能な重み付き平均として実現します。

    このノートブックでは数式を1ステップずつ、行列の中身を見ながら追いかけます。
    次のノートブック（ミニGPT）でこれを組み立てて言語モデルを作ります。
    """)
    return (mo,)


# ============================================================
# インポート
# ============================================================
@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return F, np, plt, torch


# ============================================================
# Section 1: トークンと埋め込み
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. 材料：トークン列と埋め込みベクトル

    文はまず **トークン**（単語や文字）に分割され、各トークンは
    **埋め込みベクトル**（$d$ 次元の実数ベクトル）に変換されます。

    ここでは例文「**猫 が 魚 を 食べ た**」（6トークン）を使います。
    埋め込みは本来学習で得られますが、ここでは乱数で代用します
    （仕組みの理解には十分です）。

    $$
    X \in \mathbb{R}^{T \times d} \qquad T=6 \text{（トークン数）}, \; d=16 \text{（埋め込み次元）}
    $$

    行 $X[i]$ が $i$ 番目のトークンのベクトルです（0始まり）。
    """)
    return


@app.cell
def _(torch):
    TOKENS = ["猫", "が", "魚", "を", "食べ", "た"]
    T_len, d_model = len(TOKENS), 16

    torch.manual_seed(42)
    X_embed = torch.randn(T_len, d_model)
    print(f"X の形: {tuple(X_embed.shape)}  （{T_len}トークン × {d_model}次元）")
    return TOKENS, T_len, X_embed, d_model


# ============================================================
# Section 2: Q, K, V
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Q・K・V：検索エンジンのアナロジー

    各トークンのベクトルから、3つの役割の異なるベクトルを**線形変換**で作ります：

    $$
    Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
    $$

    | 記号 | 名前 | 検索エンジンでいうと |
    |:---:|:---|:---|
    | $Q$（Query） | 問い合わせ | 「私はこういう情報が欲しい」という検索キーワード |
    | $K$（Key） | 索引 | 「私はこういう情報を持っている」というWebページのタイトル |
    | $V$（Value） | 中身 | 実際に持ち帰るWebページの本文 |

    トークン $i$ は自分の $Q[i]$ と全トークンの $K[j]$ の**内積**で「関連度」を測り、
    関連度に応じて $V[j]$ を混ぜて持ち帰ります。

    $W_Q, W_K, W_V$ は**学習されるパラメータ**です。つまり
    「何を検索し、何を索引として出すか」自体をデータから学びます。
    """)
    return


@app.cell
def _(X_embed, d_model, torch):
    d_k = 8  # Q・Kの次元（d_modelより小さくてよい）

    torch.manual_seed(0)
    W_Q = torch.randn(d_model, d_k) / d_model**0.5
    W_K = torch.randn(d_model, d_k) / d_model**0.5
    W_V = torch.randn(d_model, d_k) / d_model**0.5

    Q_mat = X_embed @ W_Q
    K_mat = X_embed @ W_K
    V_mat = X_embed @ W_V
    print(f"Q: {tuple(Q_mat.shape)}, K: {tuple(K_mat.shape)}, V: {tuple(V_mat.shape)}")
    return K_mat, Q_mat, V_mat, d_k


# ============================================================
# Section 3: スコアとsoftmax
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Attentionスコア：たった1つの式

    $$
    \text{Attention}(Q, K, V) = \underbrace{\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{\text{Attention重み } A \; (T \times T)} V
    $$

    ステップ分解：

    1. **$QK^\top$**：全トークンペアの内積 → $(T, T)$ の「関連度スコア表」
       $S[i,j]$ = 「トークン $i$ がトークン $j$ をどれだけ気にするか（の生スコア）」
    2. **$\div \sqrt{d_k}$**：次元が大きいと内積が大きくなりすぎ、softmaxが
       一極集中（勾配消失）するのを防ぐスケーリング
    3. **softmax**（各行ごと）：スコアを合計1の確率分布に変換
    4. **$\times V$**：その確率で $V$ を重み付き平均 → 各トークンの新しい表現

    下のヒートマップで $A$ の中身を見てみましょう。**行が「見る側」、列が「見られる側」**です。
    """)
    return


@app.cell
def _(F, K_mat, Q_mat, V_mat, d_k):
    scores = Q_mat @ K_mat.T / d_k**0.5   # (T, T) 関連度スコア
    A_weights = F.softmax(scores, dim=-1)  # 各行を確率分布に
    out_attn = A_weights @ V_mat           # 重み付き平均で新しい表現

    print(f"スコア S: {tuple(scores.shape)}")
    print(f"重み A の各行の合計（=1になるはず）: {A_weights.sum(dim=-1).tolist()}")
    return (A_weights,)


@app.cell
def _(A_weights, TOKENS, T_len, np, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _im = _ax.imshow(A_weights.numpy(), cmap="Blues", vmin=0)
    _ax.set_xticks(np.arange(T_len), TOKENS)
    _ax.set_yticks(np.arange(T_len), TOKENS)
    _ax.set_xlabel("見られる側（Key）")
    _ax.set_ylabel("見る側（Query）")
    _ax.set_title("Attention重み行列 A（各行の合計=1）")
    for _i in range(T_len):
        for _j in range(T_len):
            _v = A_weights[_i, _j].item()
            _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center",
                     color="white" if _v > 0.5 else "black", fontsize=9)
    plt.colorbar(_im)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **注意**：この重みは乱数の $W_Q, W_K, W_V$ から計算したものなので、
    > 言語的な意味はまだありません。**学習すると**「食べ」の行で「猫」（主語）や
    > 「魚」（目的語）の重みが大きくなる、といった構造が現れます。
    > ここで見るべきは「行列演算だけで、トークン間の参照が実現される」という仕組みです。
    """)
    return


# ============================================================
# Section 4: 温度（スケーリング）の効果
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. なぜ $\sqrt{d_k}$ で割るのか：スライダーで体感

    softmaxは入力の**スケール**に敏感です。スコアを大きくすると分布が一極集中
    （ほぼone-hot）になり、勾配がほとんど流れなくなります。

    スコア倍率を動かして、Attention重みの分布がどう変わるか見てください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    scale_slider = mo.ui.slider(0.1, 10, value=1.0, step=0.1, label="スコア倍率")
    scale_slider
    return (scale_slider,)


@app.cell
def _(F, K_mat, Q_mat, TOKENS, T_len, d_k, np, plt, scale_slider):
    _scaled = F.softmax(Q_mat @ K_mat.T / d_k**0.5 * scale_slider.value, dim=-1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _ax1.imshow(_scaled.numpy(), cmap="Blues", vmin=0, vmax=1)
    _ax1.set_xticks(np.arange(T_len), TOKENS)
    _ax1.set_yticks(np.arange(T_len), TOKENS)
    _ax1.set_title(f"倍率 {scale_slider.value:.1f} のAttention重み")

    _row = 4  # 「食べ」の行
    _ax2.bar(TOKENS, _scaled[_row].numpy(), color="C0")
    _ax2.set_ylim(0, 1)
    _ax2.set_title(f"「{TOKENS[_row]}」が各トークンに向ける重み")
    _ax2.set_ylabel("Attention重み")
    _ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 倍率を**大きく**すると → ほぼ1点集中（one-hot）。勾配が消えて学習が止まる
    - 倍率を**小さく**すると → ほぼ一様分布。「どこも同じだけ見る」＝情報の選別ができない

    $\sqrt{d_k}$ で割るのは、次元 $d_k$ が大きいほど内積の分散が $d_k$ に比例して
    大きくなるため、それを打ち消して**ちょうどよいコントラスト**に保つためです。
    """)
    return


# ============================================================
# Section 5: 因果マスク
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. 因果マスク：未来を見てはいけない

    GPTのような**言語モデル**は「次のトークンを予測する」タスクで学習します。
    このとき、トークン $i$ の予測に $i$ より**後ろ**のトークンが見えてしまったら
    カンニングです。

    そこで、スコア行列の**右上三角**（未来との関連度）を $-\infty$ に置き換えてから
    softmaxします。$e^{-\infty} = 0$ なので、未来への重みはちょうど0になります。

    ```python
    mask = torch.tril(torch.ones(T, T))          # 下三角が1
    scores = scores.masked_fill(mask == 0, float("-inf"))
    ```
    """)
    return


@app.cell
def _(F, K_mat, Q_mat, TOKENS, T_len, d_k, np, plt, torch):
    _scores = Q_mat @ K_mat.T / d_k**0.5
    causal_mask = torch.tril(torch.ones(T_len, T_len))
    _masked_scores = _scores.masked_fill(causal_mask == 0, float("-inf"))
    A_causal = F.softmax(_masked_scores, dim=-1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for _ax, _A, _title in [
        (_ax1, F.softmax(_scores, dim=-1), "マスクなし（双方向）\nBERTなどのエンコーダ型"),
        (_ax2, A_causal, "因果マスクあり\nGPTなどのデコーダ型"),
    ]:
        _ax.imshow(_A.numpy(), cmap="Blues", vmin=0)
        _ax.set_xticks(np.arange(T_len), TOKENS)
        _ax.set_yticks(np.arange(T_len), TOKENS)
        _ax.set_title(_title)
        for _i in range(T_len):
            for _j in range(T_len):
                _v = _A[_i, _j].item()
                _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center",
                         color="white" if _v > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    右の図では、各トークンは**自分と過去だけ**を見ています（右上が全部0）。
    1行目「猫」は自分しか見えないので重みが1.00になっている点にも注目してください。
    """)
    return


# ============================================================
# Section 6: 位置エンコーディング
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. 位置情報：Attentionは語順を知らない

    実はAttentionには重大な弱点があります。$QK^\top$ は**内積の表**にすぎないので、
    トークンの並び順を入れ替えても（行と列が入れ替わるだけで）各トークンが受け取る
    情報は変わりません。つまり **「猫が魚を食べた」と「魚が猫を食べた」を区別できない** のです。

    そこで、埋め込みに**位置ごとに異なるベクトル**を足してから入力します：

    $$
    X' = X + P, \qquad P[t, 2i] = \sin\!\frac{t}{10000^{2i/d}}, \quad P[t, 2i+1] = \cos\!\frac{t}{10000^{2i/d}}
    $$

    （原論文のsin/cos版。GPTでは $P$ 自体を学習パラメータにする方式が使われ、
    次のノートブックではそちらを採用します）
    """)
    return


@app.cell
def _(np, plt, torch):
    _T, _d = 64, 32
    _pos = torch.arange(_T).unsqueeze(1).float()
    _i = torch.arange(0, _d, 2).float()
    _angle = _pos / (10000 ** (_i / _d))
    P_enc = torch.zeros(_T, _d)
    P_enc[:, 0::2] = torch.sin(_angle)
    P_enc[:, 1::2] = torch.cos(_angle)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _im = _ax1.imshow(P_enc.numpy().T, cmap="RdBu_r", aspect="auto")
    _ax1.set_xlabel("位置 t")
    _ax1.set_ylabel("次元")
    _ax1.set_title("位置エンコーディング P（各列＝その位置の「指紋」）")
    plt.colorbar(_im, ax=_ax1)

    for _dim in [0, 4, 12]:
        _ax2.plot(np.arange(_T), P_enc[:, _dim].numpy(), label=f"次元 {_dim}")
    _ax2.set_xlabel("位置 t")
    _ax2.set_ylabel("値")
    _ax2.set_title("次元ごとに異なる周期の波 → 位置が一意に決まる")
    _ax2.legend()
    _ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


# ============================================================
# Section 7: マルチヘッド
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. マルチヘッドAttention：複数の「観点」で同時に見る

    1組の $(W_Q, W_K, W_V)$ が学べる参照パターンは1種類だけです。
    そこで、独立な $(W_Q, W_K, W_V)$ を $h$ 組用意して**並列に**Attentionを計算し、
    結果を連結します。これが**マルチヘッド**です。

    - あるヘッドは「直前のトークン」を見る係
    - あるヘッドは「文の主語」を見る係
    - あるヘッドは「同じ単語の繰り返し」を見る係 …

    のように、**役割分担**が自然に生まれます（学習後のGPTで実際に観察されています）。

    下は4ヘッド分のAttention重み（乱数初期化）。同じ入力でも
    ヘッドごとに**まったく違うパターン**で参照していることを確認してください。
    """)
    return


@app.cell
def _(F, TOKENS, T_len, X_embed, d_model, np, plt, torch):
    _n_heads, _d_head = 4, 4
    _fig, _axes = plt.subplots(1, _n_heads, figsize=(13, 3.2))
    torch.manual_seed(7)
    for _h, _ax in enumerate(_axes):
        _wq = torch.randn(d_model, _d_head) / d_model**0.5
        _wk = torch.randn(d_model, _d_head) / d_model**0.5
        _A = F.softmax((X_embed @ _wq) @ (X_embed @ _wk).T / _d_head**0.5, dim=-1)
        _ax.imshow(_A.numpy(), cmap="Blues", vmin=0)
        _ax.set_xticks(np.arange(T_len), TOKENS, fontsize=7)
        _ax.set_yticks(np.arange(T_len), TOKENS, fontsize=7)
        _ax.set_title(f"ヘッド {_h}", fontsize=10)
    plt.suptitle("マルチヘッド：ヘッドごとに異なる参照パターン")
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

    | 部品 | 役割 |
    |:---|:---|
    | 埋め込み $X$ | トークンをベクトルにする |
    | $Q=XW_Q,\; K=XW_K,\; V=XW_V$ | 「検索クエリ／索引／中身」を学習可能な変換で作る |
    | $\text{softmax}(QK^\top/\sqrt{d_k})$ | どのトークンをどれだけ見るかの確率分布 |
    | $\times V$ | 重み付き平均で文脈を取り込んだ新しい表現を作る |
    | 因果マスク | 未来のトークンを見えなくする（GPT型に必須） |
    | 位置エンコーディング | 語順の情報を注入する |
    | マルチヘッド | 複数の参照パターンを並列に学習する |

    ### 確認クイズ

    1. Attention重み行列 $A$ の形が $(T, T)$ になるのはなぜか？各行の合計が1になるのはなぜか？
    2. 因果マスクで $-\infty$ を**softmaxの前**に入れるのはなぜか？（0を掛けるのではだめ？）
    3. 位置エンコーディングがないと「猫が魚を食べた」と「魚が猫を食べた」が区別できないのはなぜか？

    ### 次のステップ

    **`mini_gpt_marimo.py`** で、今日の部品（Attention + マスク + 位置埋め込み + マルチヘッド）を
    組み立てて、シェイクスピア風の文章を生成する言語モデルをゼロから学習させます。
    """)
    return


if __name__ == "__main__":
    app.run()
