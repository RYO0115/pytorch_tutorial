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
    # ミニGPTをゼロから作る：文字レベル言語モデル

    前のノートブック（`attention_basics_marimo.py`）で学んだ部品を組み立てて、
    **シェイクスピア風の文章を生成する小さなGPT** を作ります
    （Andrej Karpathy の "Let's build GPT" と同じ流儀です）。

    作るものの全体像：

    1. **トークナイザ**：文字 ↔ 整数の変換（文字レベルなので超シンプル）
    2. **ベースライン**：Bigramモデル（直前の1文字だけで次を予測）
    3. **本命**：Transformerブロック（Attention + FFN）を積んだミニGPT（約80万パラメータ）
    4. **生成**：温度パラメータ付きサンプリング

    データは `data/tinyshakespeare.txt`（シェイクスピア戯曲、約100万文字）です。
    """)
    return (mo,)


# ============================================================
# インポートとデバイス
# ============================================================
@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"使用デバイス: {device}")
    return F, device, nn, plt, torch


# ============================================================
# Section 1: データとトークナイザ
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. データとトークナイザ

    言語モデルの入力は整数の列です。ここでは**文字レベル**、つまり
    1文字を1トークンとします。語彙は「テキストに登場する文字の集合」だけ。

    - `encode("hi")` → `[46, 47]` のように文字を整数へ
    - `decode([46, 47])` → `"hi"` のように整数を文字へ

    実際のGPTはBPE（Byte Pair Encoding）でサブワード単位に分割しますが、
    仕組みの学習には文字レベルで十分です。
    """)
    return


@app.cell
def _(torch):
    with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as _f:
        text = _f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data_all = torch.tensor(encode(text), dtype=torch.long)
    # 90% を訓練、10% を検証に
    _n = int(0.9 * len(data_all))
    train_data = data_all[:_n]
    val_data = data_all[_n:]

    print(f"総文字数: {len(text):,}")
    print(f"語彙サイズ: {vocab_size}（登場する文字の種類）")
    print(f"語彙: {''.join(chars[1:])}")
    print(f"\nencode('hello') = {encode('hello')}")
    print(f"decode(...) = '{decode(encode('hello'))}'")
    print(f"\n冒頭:\n{text[:150]}")
    return decode, encode, train_data, val_data, vocab_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 学習データの作り方：「次の1文字」を当てるだけ

    長さ `block_size` の文字列を切り出すと、**1つずつずらした**入力と正解のペアが作れます。

    ```
    入力 x:  "To be or no"
    正解 y:  "o be or not"      ← xを1文字左にずらしたもの
    ```

    つまり位置 $t$ では「$x[0..t]$ を見て $y[t]$（次の文字）を予測する」問題が
    `block_size` 個同時に得られます。ラベル付け作業ゼロでデータが無限に作れる——
    これが**自己教師あり学習**です。
    """)
    return


@app.cell
def _(decode, torch, train_data, val_data):
    block_size = 128  # 一度に見る文脈の長さ（トークン数）
    batch_size = 64

    def get_batch(split, generator=None):
        _data = train_data if split == "train" else val_data
        _ix = torch.randint(len(_data) - block_size - 1, (batch_size,), generator=generator)
        _x = torch.stack([_data[i : i + block_size] for i in _ix])
        _y = torch.stack([_data[i + 1 : i + block_size + 1] for i in _ix])
        return _x, _y

    _g = torch.Generator().manual_seed(0)
    _xb, _yb = get_batch("train", _g)
    print(f"入力バッチ x: {tuple(_xb.shape)}  正解バッチ y: {tuple(_yb.shape)}")
    print(f"\nx[0] の冒頭: '{decode(_xb[0][:40].tolist())}'")
    print(f"y[0] の冒頭: '{decode(_yb[0][:40].tolist())}'  ← 1文字ずれている")
    return block_size, get_batch


# ============================================================
# Section 3: Bigramベースライン
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. ベースライン：Bigramモデル

    まず**一番単純なモデル**から。直前の1文字だけを見て次の文字を予測します。
    実体は「(語彙×語彙) の表を引くだけ」＝ `nn.Embedding` 1個です。

    文脈を全く使えないので性能の上限は低いですが、
    **①学習ループ ②生成ループ** の骨格を最小構成で確認できます。
    後でTransformerに差し替えたとき、この2つのループは**一切変わりません**。
    """)
    return


@app.cell
def _(F, nn, torch):
    class BigramModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            # 行=現在の文字, 列=次の文字のlogit という表
            self.table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx):
            return self.table(idx)  # (B, T) -> (B, T, vocab_size)

    @torch.no_grad()
    def generate(model, idx, max_new_tokens, block_size=None, temperature=1.0):
        """idx (B, T) に続くトークンを1つずつサンプリングして伸ばす"""
        model.eval()
        for _ in range(max_new_tokens):
            _ctx = idx if block_size is None else idx[:, -block_size:]
            _logits = model(_ctx)[:, -1, :] / temperature  # 最後の位置だけ使う
            _probs = F.softmax(_logits, dim=-1)
            _next = torch.multinomial(_probs, num_samples=1)
            idx = torch.cat([idx, _next], dim=1)
        return idx
    return BigramModel, generate


@app.cell(hide_code=True)
def _(mo):
    bigram_btn = mo.ui.run_button(label="Bigramを学習して生成（〜30秒）")
    bigram_btn
    return (bigram_btn,)


@app.cell
def _(BigramModel, bigram_btn, decode, device, generate, get_batch, mo, torch, vocab_size):
    mo.stop(not bigram_btn.value, mo.md("👆 ボタンを押すとBigramの学習と生成が走ります"))

    torch.manual_seed(0)
    _bigram = BigramModel(vocab_size).to(device)
    _opt = torch.optim.AdamW(_bigram.parameters(), lr=1e-2)

    _prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # 改行文字から開始
    print("--- 学習前の生成（完全なランダム文字列） ---")
    print(decode(generate(_bigram, _prompt, 150)[0].tolist()))

    _bigram.train()
    for _step in range(3000):
        _x, _y = get_batch("train")
        _x, _y = _x.to(device), _y.to(device)
        _logits = _bigram(_x)
        _loss = torch.nn.functional.cross_entropy(
            _logits.view(-1, vocab_size), _y.view(-1)
        )
        _opt.zero_grad()
        _loss.backward()
        _opt.step()

    print(f"\n--- 3000ステップ学習後（loss={_loss.item():.3f}）の生成 ---")
    print(decode(generate(_bigram, _prompt, 300)[0].tolist()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    学習後は「英語っぽい文字の並び」（母音と子音の交互、単語の長さ）にはなりますが、
    単語も文法も成立しません。**直前の1文字しか見ていない**のだから当然です。
    → 文脈を見るために **Attention** の出番です。
    """)
    return


# ============================================================
# Section 4: ミニGPT本体
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. ミニGPTを組み立てる

    前のノートブックの部品を `nn.Module` にします。構成は本物のGPTの縮小版です：

    ```
    トークン埋め込み + 位置埋め込み（学習型）
      ↓
    Transformerブロック × 4        ┐
      ├ Multi-Head Attention（因果マスク付き, 4ヘッド）│ 残差接続 +
      └ FeedForward（4倍に広げて戻すMLP）             │ LayerNorm（Pre-LN）
      ↓                                              ┘
    LayerNorm → Linear(n_embd → vocab_size)  = 次の文字のlogits
    ```

    新登場の部品は2つだけ：

    - **残差接続** `x = x + f(x)`：深く積んでも勾配が流れる高速道路
    - **LayerNorm**：各トークンのベクトルを正規化して学習を安定させる
    """)
    return


@app.cell
def _(F, block_size, nn, torch, vocab_size):
    n_embd, n_head, n_layer, dropout = 128, 4, 4, 0.1

    class CausalSelfAttention(nn.Module):
        """マルチヘッド因果Attention（全ヘッドを1つの行列演算でまとめて計算）"""

        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(n_embd, 3 * n_embd)  # Q,K,Vを一括生成
            self.proj = nn.Linear(n_embd, n_embd)
            self.drop = nn.Dropout(dropout)
            _mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", _mask.view(1, 1, block_size, block_size))

        def forward(self, x):
            B, T, C = x.shape
            _q, _k, _v = self.qkv(x).split(n_embd, dim=2)
            # (B, T, C) -> (B, n_head, T, head_dim) にヘッドを分割
            _hd = C // n_head
            _q = _q.view(B, T, n_head, _hd).transpose(1, 2)
            _k = _k.view(B, T, n_head, _hd).transpose(1, 2)
            _v = _v.view(B, T, n_head, _hd).transpose(1, 2)

            _att = _q @ _k.transpose(-2, -1) / _hd**0.5      # (B, nh, T, T)
            _att = _att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            _att = F.softmax(_att, dim=-1)
            _out = _att @ _v                                  # (B, nh, T, hd)
            _out = _out.transpose(1, 2).contiguous().view(B, T, C)  # ヘッドを連結
            return self.drop(self.proj(_out))

    class FeedForward(nn.Module):
        """各トークン独立に適用されるMLP（Attentionで集めた情報を加工する係）"""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """Transformerブロック：Attention + FFN、それぞれ残差接続とPre-LayerNorm"""

        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(n_embd)
            self.ffn = FeedForward()

        def forward(self, x):
            x = x + self.attn(self.ln1(x))  # 残差：元のxに「読み取った文脈」を足す
            x = x + self.ffn(self.ln2(x))
            return x

    class MiniGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Embedding(block_size, n_embd)  # 学習型の位置埋め込み
            self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx):
            B, T = idx.shape
            _tok = self.tok_emb(idx)                                   # (B, T, C)
            _pos = self.pos_emb(torch.arange(T, device=idx.device))   # (T, C)
            x = _tok + _pos
            x = self.blocks(x)
            x = self.ln_f(x)
            return self.head(x)                                        # (B, T, vocab)

    _n_params = sum(p.numel() for p in MiniGPT().parameters())
    print(f"ミニGPTのパラメータ数: {_n_params:,}")
    print(f"設定: 埋め込み{n_embd}次元, {n_head}ヘッド × {n_layer}層, 文脈長{block_size}")
    return (MiniGPT,)


# ============================================================
# Section 5: 学習
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. 学習

    学習ループはBigramのときと**1行も変わりません**（モデルを差し替えただけ）。
    損失は「次の文字の分類問題」としてのクロスエントロピーです。

    - 学習前の損失の目安：$-\log(1/65) \approx 4.17$（65文字から当てずっぽう）
    - 2,000ステップ（MPSで3分前後）で検証損失 ≈ 1.7 まで下がり、
      単語・改行・登場人物名らしきものが現れます
    - 時間があれば5,000ステップまで伸ばすと質がさらに上がります
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    steps_slider = mo.ui.slider(500, 5000, value=2000, step=500, label="学習ステップ数")
    train_btn = mo.ui.run_button(label="ミニGPTを学習（2000ステップ ≈ 3分）")
    mo.vstack([steps_slider, train_btn])
    return steps_slider, train_btn


@app.cell
def _(MiniGPT, device, get_batch, mo, steps_slider, torch, train_btn, vocab_size):
    mo.stop(not train_btn.value, mo.md("👆 ボタンを押すと学習が始まります"))

    torch.manual_seed(0)
    gpt = MiniGPT().to(device)
    _opt = torch.optim.AdamW(gpt.parameters(), lr=3e-4)

    @torch.no_grad()
    def _estimate_loss():
        gpt.eval()
        _out = {}
        for _split in ["train", "val"]:
            _losses = torch.zeros(10)
            for _k in range(10):
                _x, _y = get_batch(_split)
                _x, _y = _x.to(device), _y.to(device)
                _logits = gpt(_x)
                _losses[_k] = torch.nn.functional.cross_entropy(
                    _logits.view(-1, vocab_size), _y.view(-1)
                ).item()
            _out[_split] = _losses.mean().item()
        gpt.train()
        return _out

    _steps = steps_slider.value
    _eval_every = max(_steps // 10, 100)
    loss_history = {"step": [], "train": [], "val": []}

    gpt.train()
    with mo.status.progress_bar(total=_steps, title="ミニGPT学習中...") as _bar:
        for _step in range(_steps):
            if _step % _eval_every == 0:
                _est = _estimate_loss()
                loss_history["step"].append(_step)
                loss_history["train"].append(_est["train"])
                loss_history["val"].append(_est["val"])
            _x, _y = get_batch("train")
            _x, _y = _x.to(device), _y.to(device)
            _logits = gpt(_x)
            _loss = torch.nn.functional.cross_entropy(
                _logits.view(-1, vocab_size), _y.view(-1)
            )
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            _bar.update()

    _final = _estimate_loss()
    loss_history["step"].append(_steps)
    loss_history["train"].append(_final["train"])
    loss_history["val"].append(_final["val"])
    torch.save(gpt.state_dict(), "data/mini_gpt.pth")
    print(f"最終損失: 訓練 {_final['train']:.3f} / 検証 {_final['val']:.3f}")
    print("→ 'data/mini_gpt.pth' に保存しました")
    return gpt, loss_history


@app.cell
def _(loss_history, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(loss_history["step"], loss_history["train"], "o-", label="訓練損失")
    _ax.plot(loss_history["step"], loss_history["val"], "s-", label="検証損失")
    _ax.axhline(4.17, color="gray", ls=":", label="当てずっぽう（-log(1/65)）")
    _ax.set_xlabel("学習ステップ")
    _ax.set_ylabel("クロスエントロピー損失")
    _ax.set_title("ミニGPTの学習曲線")
    _ax.legend()
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.gca()
    return


# ============================================================
# Section 6: 生成
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. 生成：温度で「堅実さ」と「冒険心」を調整

    生成時はlogitsを**温度** $\tau$ で割ってからsoftmaxします：
    $p_i = \text{softmax}(z_i / \tau)$

    - $\tau < 1$：分布が尖る → 確率の高い文字ばかり選ぶ（堅実・単調）
    - $\tau = 1$：学習した分布のまま
    - $\tau > 1$：分布が平らになる → 意外な文字も選ぶ（多様・破綻しやすい）

    ChatGPTのAPIにある `temperature` パラメータはまさにこれです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    temp_slider = mo.ui.slider(0.3, 2.0, value=0.8, step=0.1, label="温度 τ")
    prompt_text = mo.ui.text(value="ROMEO:", label="プロンプト（英語）")
    gen_btn = mo.ui.run_button(label="500文字を生成")
    mo.vstack([prompt_text, temp_slider, gen_btn])
    return gen_btn, prompt_text, temp_slider


@app.cell
def _(
    block_size,
    decode,
    device,
    encode,
    gen_btn,
    generate,
    gpt,
    mo,
    prompt_text,
    temp_slider,
    torch,
):
    mo.stop(not gen_btn.value, mo.md("👆 学習後、ボタンで文章を生成します"))

    _ids = torch.tensor([encode(prompt_text.value)], dtype=torch.long, device=device)
    _out = generate(
        gpt, _ids, max_new_tokens=500,
        block_size=block_size, temperature=temp_slider.value,
    )
    mo.md(f"**生成結果（温度 {temp_slider.value}）：**\n```\n{decode(_out[0].tolist())}\n```")
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
    | 自己教師あり学習 | 「1文字ずらす」だけでラベル付きデータが無限に作れる |
    | Bigram → GPT | 学習・生成ループはそのまま、モデルだけ差し替え |
    | Transformerブロック | Attention（情報を集める）+ FFN（加工する）+ 残差 + LayerNorm |
    | 因果マスク | 学習時に全位置の予測を並列計算しつつカンニングを防ぐ |
    | 温度サンプリング | logits ÷ τ で生成の多様性を制御 |

    本物のGPTとの違いは本質的には**規模だけ**です：
    パラメータ 80万 → 数千億、文脈長 128 → 数十万、文字 → BPEトークン、
    そして事後学習（RLHF）。仕組みの骨格は今日作ったものと同じです。

    ### 確認クイズ

    1. 学習時、長さ $T$ の系列から**いくつの**予測問題が同時に得られるか？
    2. 残差接続 `x = x + attn(ln(x))` を `x = attn(ln(x))` にすると何が起きるか？
    3. 温度 τ→0 の極限では生成はどうなるか？

    ### 発展課題

    - `n_layer` や `n_embd` を増やして検証損失がどこまで下がるか実験する（過学習にも注意）
    - 学習済みの `gpt` のAttention重みを取り出し、前のノートブックのようにヒートマップ表示する
    - 日本語テキスト（青空文庫など）に差し替えて学習してみる（語彙数が数千になる点に注意）
    - Karpathyの動画 ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) と
      [nanoGPT](https://github.com/karpathy/nanoGPT) で答え合わせをする
    """)
    return


if __name__ == "__main__":
    app.run()
