# PyTorch Tutorial Project

## 環境
- Python環境: `pytorch-learning/.venv`（uvで管理）
- marimoでインタラクティブなノートブックを作成する
- ノートブックの配置先: `pytorch-learning/src/basic/<トピック>/`

## 数学・行列演算の解説を求められたとき

**数学概念・行列演算・線形代数について解説を求められた場合は、必ず最初に `skill.md` を読み込み、そのルールとテンプレートに従って解説すること。**

トリガーとなる依頼の例：
- 「〜を教えてください」「〜を説明してください」
- 「〜とは何か」「〜の仕組みを理解したい」
- 「〜を可視化してください」「marimoで〜を説明して」
- 行列積・内積・外積・転置・逆行列・固有値・SVD・微分・勾配などのキーワード

読み込むファイル: `skill.md`（プロジェクトルート）

## marimoノートブックのルール

- 日本語フォントは必ず設定する
  ```python
  plt.rcParams["font.family"] = "Hiragino Sans"
  plt.rcParams["axes.unicode_minus"] = False
  ```
- グラフタイトル・軸ラベルは日本語で書く
- インタラクティブコントロール（`mo.ui.slider` 等）を積極的に使う
- 起動コマンド: `cd pytorch-learning && .venv/bin/marimo run src/basic/<file>.py`

## 既存テンプレート
- `tutorial_explain_skill.md`: PyTorchチュートリアルのMarkdown解説作成ルール
- `skill.md`: 数学・行列演算の解説テンプレート（marimo可視化付き）
