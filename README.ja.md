# Admetus

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)

[English](README.md) | **日本語**

Admetus は SMILES から ADMET 特性を予測します。公開データセットごとに LightGBM の fold
アンサンブルを 1 つ学習し、特徴量は Morgan フィンガープリントと 2D ファーマコフォアシグネチャの
連結です。CLI は人間よりもプログラムが使うことを前提にしており、自己記述的・非対話・終了コードが
安定しています。

SMILES の列を持つ CSV を渡すと予測列を追記します。それ以外の列は、受け取ったままバイト単位で
変えずに返します。

> **状態: バージョン 0.1、開発中。** 学習済みモデルはまだ公開していないので、下の精度の表は
> 仮の数値で埋めるのではなく意図的に空にしています。`admetus train` でローカルに学習してください。
> インターフェースは今後変わることがあります。

- [特長](#特長)
- [仕組み](#仕組み)
- [必要なもの](#必要なもの)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [CLI リファレンス](#cli-リファレンス)
- [出力列](#出力列)
- [エンドポイント](#エンドポイント)
- [ベンチマーク](#ベンチマーク)
- [設定](#設定)
- [制限](#制限)
- [開発](#開発)
- [参考文献](#参考文献)
- [ライセンス](#ライセンス)

## 特長

- **29 エンドポイント**、公開データセットごとに 1 モデル。Therapeutics Data Commons の ADMET
  ベンチマークグループの 22 データセットと、さらに 7 件の公開 ADMET データセット。
- **他の列には手を付けません。** 引用符・改行コード・BOM・文字エンコーディング・列名の重複・
  列順はすべてそのまま残ります。解釈するのは SMILES 列だけです。
- **失敗した分子は報告し、別の分子に置き換えません。** パースできない入力は予測セルが空になり
  理由が入ります。実行自体は失敗になりません。
- **ハイパーパラメータの自動チューニング**を、モデルを交差検証するのと同じ scaffold group 分割の
  上で行います。study は SQLite に保存され再開できます。
- **数値に誠実であること。** 報告する指標にはすべてブートストラップ信頼区間を付け、交差検証スコアは
  偏りがあるものとして記録し性能としては報告せず、小規模データセットには過大に読まないための印を
  付けます。
- **プログラムのための設計。** `admetus endpoints --json` が全エンドポイントと出力列を説明するので、
  呼び出し側はこのファイルを読まずに呼び出しを組み立てられます。`admetus doctor` は 1 回の呼び出しで
  「予測できるか、できないなら何を直すか」を答えます。

## 仕組み

エンドポイントごとに個別に学習します。データセット間で化合物も単位もアッセイも共有していないためです。

1. **標準化してから分割する。** 分割の前に SMILES を整え、塩を除き、電荷を中和します。同じ分子に
   潰れる 2 行が分割の両側に残らないようにするためです。
2. **特徴量化する。** Morgan フィンガープリントと 2D ファーマコフォアシグネチャを連結します。
   ファーマコフォア側は 2 点または 3 点のファーマコフォア特徴の組と各ペアのトポロジカル距離を
   列挙するので、1 ビットが「ドナーとアクセプターが 4〜5 結合離れ、芳香環が両方から 6〜7 結合」
   のような意味を持ちます。
3. **scaffold で分割する。** scaffold group をサイズの大きい順に学習側へ詰めるので、珍しい化学構造が
   意図的にホールドアウト側に集まります。ランダム分割では近い類縁体が両側に入り、モデルを
   過大評価してしまいます。
4. **モデルから見える粒度で分割を監査する。** どちらのフィンガープリントも立体化学を見ないので、
   鏡像異性体のペアはモデルにとって同一の分子です。特徴ベクトルで各行をハッシュし、問題のある
   **学習側**の行を落とします。ホールドアウト側は残すので、指標は公開ベンチマークと同じ化合物の
   上で測られ続けます。
5. **列を枝刈りする。** 学習分子のうち立っている数が少なすぎる列・多すぎる列を落とします。基準は
   件数ではなく出現率です。固定の件数だと最小のデータセットでは「分子の 1.8%」、最大のものでは
   「0.04%」を意味してしまいます。
6. **チューニングしてから再学習する。** ハイパーパラメータは自動チューナーが決め、配布するモデルは
   そのあと固定のブースティング回数で再学習します。採点に使ったデータで打ち切られたモデルが
   そのまま配布されないようにするためです。
7. **校正して測る。** out-of-fold 予測から確率の校正・推奨決定閾値・conformal 区間の幅を決めます。
   ホールドアウトセットを見るのはちょうど 1 回だけです。

## 必要なもの

- Python 3.12 または 3.13
- OpenMP ランタイム。LightGBM の wheel はこれにリンクしていますが同梱していないため、無いと
  最初の予測が何も説明しない `dlopen` エラーで失敗します。`admetus doctor` が直し方を示します。
- macOS (arm64 または x86_64) もしくは Linux (amd64 または arm64)
- 全エンドポイントで予測する場合、メモリ 3 GB 程度

## インストール

macOS では先に OpenMP ランタイムを入れます。

```bash
brew install libomp
```

Debian / Ubuntu では次のとおりです。

```bash
sudo apt-get install -y libgomp1
```

そのうえでパッケージを入れます。

```bash
uv pip install -e '.[train]'
```

実際に予測できる状態かを確認します。

```bash
admetus doctor
```

### Docker

イメージには OpenMP ランタイムが入っているので、ほかに何も必要ありません。

```bash
docker run --rm -v "$PWD:/data" ghcr.io/ytworks/admetus:latest endpoints
```

予測にはバンドルも必要です。イメージは `/models` から読むので、バンドルのあるディレクトリを
マウントしてください。`/data` だけをマウントすると `/models` は空のボリュームのままで、
モデルが見つかりません。

```bash
docker run --rm -v "$PWD:/data" -v "$PWD/models:/models" ghcr.io/ytworks/admetus:latest predict -i /data/in.csv -o /data/out.csv --endpoint dili
```

## クイックスタート

予測できるものと、どのモデルが手元にあるかを一覧します。

```bash
admetus endpoints
```

データセットを取得してモデルを学習します。各データセットの digest はレジストリにピン留めしてあるので、
一致しないファイルは採用せず拒否します。

```bash
admetus data fetch --endpoint dili
```

```bash
admetus train --endpoint dili --time-budget 600
```

ファイルを使わず 1 構造だけ予測します。

```bash
admetus predict --smiles 'CC(=O)Oc1ccccc1C(=O)O' --endpoint dili --json
```

CSV に対して予測し、列を追記してレポートを書き出します。

```bash
admetus predict -i compounds.csv -o predicted.csv --endpoint dili --report report.json
```

全エンドポイントを学習します。数時間かかり、既にモデルがあるものはスキップし、再開できます。

```bash
admetus train --all --time-budget 900
```

公式プロトコルを実行します。ベンチマークグループが公開している分割の学習側で学習し、その
テストセットをシードごとに 1 回だけ採点します。リーダーボードと比較できるのはこの数値で、
`admetus evaluate` が報告するのは本プロジェクト自身の scaffold 分割です。

```bash
admetus benchmark fetch
```

```bash
admetus benchmark run --all --time-budget 300 --report official.json
```

手元のバンドルをリリース用にまとめ、公開済みのものを導入します。

```bash
admetus models package --out dist/models
```

```bash
admetus models pull --all
```

## CLI リファレンス

ログは標準エラー、データは標準出力に出るので、出力はパイプできます。`--json` は機械可読な
オブジェクトを 1 つだけ出力します。対話的に尋ねることはありません。

| コマンド | 内容 |
|---|---|
| `admetus doctor` | この環境で予測できるか、できないなら何を直すかを答える |
| `admetus endpoints` | 全エンドポイントをタスク・指標・単位・出力列とともに一覧する |
| `admetus data fetch` | 公開データセットを取得し、ピン留めした digest と照合する |
| `admetus data info` | 正規化後のデータセットを説明する |
| `admetus train` | 1 つ以上のエンドポイントをチューニングして学習する |
| `admetus evaluate` | 学習時に記録したホールドアウト指標を報告する |
| `admetus benchmark fetch` | ベンチマークグループが公開する学習/テスト分割を取得する |
| `admetus benchmark run` | その分割の上で公式の 5 シードのプロトコルを実行する |
| `admetus predict` | CSV に予測列を追記する |
| `admetus models list` | 手元のモデルバンドルを一覧し、必要なら検証する |
| `admetus models show` | バンドルの記録をすべて出力する |
| `admetus models package` | digest のマニフェスト付きでバンドルをリリース用にまとめる |
| `admetus models pull` | 公開済みバンドルを、各アーカイブを検証しながら導入する |

エンドポイントを選ぶコマンドはすべて、繰り返し指定できる `--endpoint KEY` と `--all` を取ります。

### 終了コード

| コード | 意味 |
|---|---|
| 0 | 成功。予測が空の行があっても変わらない |
| 1 | 予期しない内部エラー |
| 2 | 使い方または入力の誤り |
| 3 | モデルバンドルが無い、または検証に失敗 |
| 4 | `--strict` 指定時のみ、または全行が失敗した場合 |
| 5 | ネットワークまたはダウンロードの失敗。再試行が効くか判断できるように区別している |
| 130 | 中断 |

### 環境変数

パスや資源を指定するフラグにはすべて対応する環境変数があります。砂箱の中の呼び出し側はコマンド行を
書き換えるより環境変数を設定するためです: `ADMETUS_MODELS_DIR`、`ADMETUS_DATA_DIR`、
`ADMETUS_CACHE_DIR`、`ADMETUS_JOBS`。

## 出力列

既定ではエンドポイントごとに 3 列を、行の末尾に追記します。

| 列 | 意味 |
|---|---|
| `<key>` | 予測値。分類では校正済みの陽性クラス確率、回帰では公開されている尺度の上での値 |
| `<key>_ens_sd` | fold アンサンブル間の不一致。log 尺度のエンドポイントでは `<key>_sd_log10` という名前になる。log 値の標準偏差は公開単位での標準偏差ではないため |
| `<key>_in_domain` | 学習データに類似度が閾値以上の近傍があるか。ヒューリスティックな判定で、導出元の類似度と誤差の関係はバンドルに記録してある |

さらに `admetus_error` 列が、分子を使えなかったときの理由を持ちます。

`--interval` は回帰エンドポイントに `<key>_lo` と `<key>_hi` を追加し、`--ad-score` は最近傍類似度
そのものを追加します。`--column-prefix` はすべてに接頭辞を付けるので、自分の列との衝突を
構造的に不可能にできます。

`<key>_ens_sd` をあえて「信頼度」と呼んでいません。これは fold ごとのモデルがどれだけ食い違うかを
測るもので、不確かさの一成分にすぎず、分類では予測値そのものの関数にほぼなってしまいます。

## エンドポイント

29 エンドポイント。official と記した 22 件が Therapeutics Data Commons の ADMET ベンチマーク
グループのデータセットです。

| 領域 | エンドポイント |
|---|---|
| 吸収 | `caco2`, `pampa`, `hia`, `pgp`, `bioavailability` |
| 分布 | `lipophilicity`, `solubility`, `bbb`, `ppbr`, `vdss` |
| 代謝 | `cyp1a2_veith`, `cyp2c9_veith`, `cyp2c19_veith`, `cyp2d6_veith`, `cyp3a4_veith`, `cyp2c9_substrate`, `cyp2d6_substrate`, `cyp3a4_substrate` |
| 排泄 | `half_life`, `clearance_hepatocyte`, `clearance_microsome` |
| 毒性 | `ld50`, `herg`, `herg_karim`, `ames`, `dili`, `carcinogens`, `skin_reaction`, `tox21_nr_ar` |

`admetus endpoints --json` が各エンドポイントの完全な説明を返します。単位・指標・追加する出力列・
ローカルにモデルがあるかどうかを含みます。

複数の列をまとめて読む前に知っておくべきことが 3 つあります。

- `herg` と `herg_karim` は同じアッセイの別キュレーションで、化合物が大きく重複しています。
  この 2 列は独立した証拠ではありません。実用の予測には 20 倍大きい `herg_karim` を、公開
  リーダーボードとの比較には `herg` を使ってください。
- CYP 阻害の 5 パネルは同じスクリーニングライブラリ由来なので、誤差も適用範囲の判定も相関します。
- `tox21_nr_ar` はアンドロゲン受容体という 1 つのアッセイで、一般的な毒性ではありません。

## ベンチマーク

29 エンドポイントすべてを学習済みです。列は 2 つあり、測っているものが違います。

- **ホールドアウト** は本プロジェクト自身の scaffold 分割で、1 度だけ参照し、ブート
  ストラップ区間付きで測った値です。最大の scaffold group を学習側に入れるため、
  ホールドアウト側には意図的に珍しい化学が残り、推定は悲観的になります。
- **公式プロトコル** はベンチマークグループが公開している分割です。その `train_val`
  側で学習し、テストセットをシードごとに 1 回採点し、プロトコルの 5 シードでの平均と
  標準偏差を示します。公開リーダーボードと比較できるのはこちらの列で、公式グループの
  22 エンドポイントを対象とします。**チューニングを切って取得**しています（5 シードは
  チューニング 5 回分のため）。したがって下限値であり、最善を尽くした値ではありません。

| Endpoint | Metric | Held-out (95% CI) | Official protocol | Notes |
|---|---|---|---|---|
| `caco2` | mae | 0.3786 [0.3344, 0.4237] | 0.3214 +/- 0.0038 |  |
| `pampa` | auroc | 0.6927 [0.6013, 0.7823] | no published split |  |
| `hia` | auroc | 0.9034 [0.7483, 0.9972] | 0.9808 +/- 0.0027 | small test set |
| `pgp` | auroc | 0.8919 [0.8462, 0.9306] | 0.9284 +/- 0.0006 |  |
| `bioavailability` | auroc | 0.6750 [0.5507, 0.7955] | 0.7259 +/- 0.0130 |  |
| `lipophilicity` | mae | 0.6633 [0.6261, 0.6990] | 0.6808 +/- 0.0008 | tuning budget-limited |
| `solubility` | mae | 1.3204 [1.2619, 1.3805] | 1.1871 +/- 0.0054 | tuning budget-limited, official run partial |
| `bbb` | auroc | 0.8731 [0.8256, 0.9135] | 0.8732 +/- 0.0019 |  |
| `ppbr` | mae | 9.1137 [7.8984, 10.3877] | 7.7789 +/- 0.0362 |  |
| `vdss` | spearman | 0.6304 [0.5308, 0.7162] | 0.6263 +/- 0.0042 |  |
| `cyp2c9_veith` | auprc | 0.7190 [0.6846, 0.7542] | 0.7228 +/- 0.0006 | tuning budget-limited |
| `cyp2d6_veith` | auprc | 0.7299 [0.6963, 0.7620] | 0.6615 +/- 0.0009 |  |
| `cyp3a4_veith` | auprc | 0.7876 [0.7625, 0.8122] | 0.8288 +/- 0.0041 | tuning budget-limited |
| `cyp1a2_veith` | auprc | 0.8661 [0.8500, 0.8820] | no published split | tuning budget-limited |
| `cyp2c19_veith` | auprc | 0.7950 [0.7711, 0.8192] | no published split | tuning budget-limited |
| `cyp2c9_substrate` | auprc | 0.2491 [0.1477, 0.3892] | 0.3627 +/- 0.0062 | interval reaches the base rate |
| `cyp2d6_substrate` | auprc | 0.5038 [0.3780, 0.6707] | 0.6895 +/- 0.0114 |  |
| `cyp3a4_substrate` | auroc | 0.6507 [0.5529, 0.7409] | 0.6537 +/- 0.0061 |  |
| `half_life` | spearman | 0.2273 [0.0576, 0.3876] | 0.5551 +/- 0.0062 |  |
| `clearance_hepatocyte` | spearman | 0.3804 [0.2533, 0.5000] | 0.4009 +/- 0.0041 |  |
| `clearance_microsome` | spearman | 0.3869 [0.2671, 0.4913] | 0.5928 +/- 0.0093 |  |
| `ld50` | mae | 0.5243 [0.4990, 0.5489] | 0.7075 +/- 0.0011 | tuning budget-limited |
| `herg` | auroc | 0.8872 [0.8238, 0.9386] | 0.8454 +/- 0.0015 |  |
| `herg_karim` | auroc | 0.8534 [0.8389, 0.8672] | no published split | tuning budget-limited |
| `ames` | auroc | 0.8025 [0.7796, 0.8244] | 0.7745 +/- 0.0009 |  |
| `dili` | auroc | 0.8342 [0.7518, 0.9119] | 0.9045 +/- 0.0035 | small test set |
| `carcinogens` | auprc | 0.8101 [0.6002, 0.9453] | no published split | small test set |
| `skin_reaction` | auroc | 0.6245 [0.4951, 0.7485] | no published split | small test set, interval reaches no signal |
| `tox21_nr_ar` | auprc | 0.5079 [0.3900, 0.6164] | no published split |  |

数値を引用する前に読むべきこと:

- **シード間の標準偏差はリーダーボードのものと比較できません。** 本実装でシードが動かす
  のは学習側内部の fold 割り当てだけで、`train_val` を train と valid に再抽出しません。
  平均は比較可能、ばらつきは比較不可です。
- **どちらの分割が難しいかはエンドポイントごとに逆向きです。** 公式分割の方が易しいのは
  `half_life`、`clearance_microsome`、`cyp2d6_substrate`、難しいのは `ld50` と `vdss` です。
- **8 エンドポイントは予算律速です。** 900 秒のチューナー予算で試行は 14〜19 にとどまり、
  小さいエンドポイントの 65〜68 に対して少ないため、この構成の上限ではなく下限です。
- **3 つの区間は無信号域に達しており**、それは欠落ではなく結果そのものです:
  `half_life`、`skin_reaction`、`cyp2c9_substrate`。
- **交差検証スコアは 29 中 26 エンドポイントで性能を過大評価**し、中央値で 0.053 でした。
  選択には使えますが、報告には使えません。

予測区間は回帰 9 エンドポイントのうち 8 つで過剰にカバーし、名目 90% に対して中央値
97.0% でした。実測カバー率は各バンドルに記録されています。公式結果のうち 1 件は部分的と
印を付けています: solubility の公開テスト 1,997 行のうち特徴量化できたのは 1,995 行で、
平均はその部分集合に対する値です。

全体の表、エンドポイントごとの選択の根拠となった特徴量セットの実測、そしてエンジニア
リング面の測定値は [docs/benchmark.md](docs/benchmark.md) にあります。

## 設定

データセットのサイズで決まる選択は、設定するのではなく計算します。29 か所で保守しなくて済むように
するためです。どの特徴量セットを使うか、列をどれだけ強く枝刈りするか、そのデータセットが指標を
点推定として読むには小さすぎるかは、すべて行数から決まります。

特徴量セットは 3 つあり、`--feature-set` で上書きできます。

| 特徴量セット | 内容 | 次元 |
|---|---|---|
| `morgan` | Morgan radius 2、2048 bit。分野標準のベースライン | 2048 |
| `morgan_p2` | Morgan radius 6 と 2 点ファーマコフォアシグネチャ | 2372 |
| `morgan_p23` | Morgan radius 6 と 2 点・3 点シグネチャ | 89852 |

学習行数が 1000 を超える場合は `morgan_p23` が既定です。それ以下では全空間は過大なので
`morgan_p2` を使い、こちらは計算も 10〜30 倍軽くなります。

## 制限

- **学習済みモデルはまだ公開していません。** ローカルで学習してください。
- **2D のみ。** コンフォマーを生成しないので、立体化学はモデルから見えません。鏡像異性体は同じ
  予測値になり、学習パイプラインはそれを取り繕わず意図的に重複として扱います。
- **`<key>_in_domain` はヒューリスティック**で、保証ではありません。似た学習分子が存在することを
  示し、バンドルがそれを期待誤差と関係づけるだけで、予測が正しいとは言っていません。
- **交差検証スコアには偏りがあり**、そのように記録しています。報告するのはホールドアウトの数値だけです。
- **小規模データセット。** 4 つのエンドポイントは化合物が 600 件未満で、テストセットが 120 分子を
  下回り、指標の区間が広くなります。これらには `low_confidence` の印が付きます。
- **探索空間は自動チューナーが持っています。** その `feature_fraction` のグリッドは 0.4 から始まり、
  非常に疎な二値行列には高すぎます。またチューナーは LightGBM 自身の希少列フィルタを無効にします。
  明示的な列枝刈りがそれを補い、エンドポイントごとに記録されます。

## 開発

```bash
uv sync --all-extras
```

```bash
uv run ruff check . && uv run pytest
```

テストはネットワークを必要としません。データセットは同梱せず取得する方式で、テストはフィクスチャを
使います。`slow` を付けたテストは小さなモデルを学習し、チューナーを通します。

```bash
uv run pytest -m 'not slow'
```

```bash
python scripts/check_readme_pair.py
```

## 参考文献

データは Therapeutics Data Commons のもので、Harvard Dataverse から取得しています。データセットの
出典・引用・ライセンスは [docs/data-sources.md](docs/data-sources.md) にまとめてあり、大半は
CC BY 4.0 です。

- Huang et al., *Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery
  and Development*, NeurIPS 2021. [arXiv:2102.09548](https://arxiv.org/abs/2102.09548)
- Huang et al., *Artificial intelligence foundation for therapeutic science*, Nature Chemical
  Biology, 2022. [doi:10.1038/s41589-022-01131-2](https://doi.org/10.1038/s41589-022-01131-2)
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
  [arXiv:1907.10902](https://arxiv.org/abs/1907.10902)
- Bemis and Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*, Journal of Medicinal
  Chemistry, 1996.
- Rogers and Hahn, *Extended-Connectivity Fingerprints*, Journal of Chemical Information and
  Modeling, 2010.
- [RDKit](https://www.rdkit.org/) を化学情報処理に使っています。2D ファーマコフォアシグネチャも含みます。

## ライセンス

Apache-2.0。[LICENSE](LICENSE) を参照してください。データセットにはそれぞれのライセンスがあり、
[docs/data-sources.md](docs/data-sources.md) にまとめています。
