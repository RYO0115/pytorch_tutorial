# PyTorch Tutorial Project

## 環境
- Python環境: `pytorch-learning/.venv`（uvで管理）
- marimoでインタラクティブなノートブックを作成する
- ノートブックの配置先: `pytorch-learning/src/basic/<トピック>/`
- marimo起動コマンド: `cd pytorch-learning && .venv/bin/marimo run src/basic/<file>.py`

## 数学・行列演算の解説を求められたとき

**数学概念・行列演算・線形代数について解説を求められた場合は、必ず最初に
`pytorch-learning/.claude/skills/math-explanation/SKILL.md` を読み込み、
そこから指示される references に従って解説すること。**

トリガーとなる依頼の例：
- 「〜を教えてください」「〜を説明してください」
- 「〜とは何か」「〜の仕組みを理解したい」
- 「〜を可視化してください」「marimoで〜を説明して」
- 行列積・内積・外積・転置・逆行列・固有値・SVD・微分・勾配などのキーワード

スキルの構成（機能ごとに分離、ディレクトリごとコピーすれば他プロジェクトへ移植可能）：
- `SKILL.md`: 入口。手順・プロジェクト固有設定（移植時はここだけ書き換える）
- `references/explanation-rules.md`: 数学・行列演算・数式解説の共通ルール
- `references/marimo-notebook.md`: marimoノートブックにまとめるルール
- `references/markdown-doc.md`: Markdown解説ドキュメントにまとめるルール
