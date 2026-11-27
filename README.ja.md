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
- **ハイパーパラメータの自動チューニング**を、モデルを交差検証するのと同じ分割の
  上で行います。study は SQLite に保存され再開できます。
- **数値に誠実であること。** 報告する指標にはすべてブートストラップ信頼区間を付け、交差検証スコアは
  偏りがあるものとして記録し性能としては報告せず、小規模データセットには過大に読まないための印を
  付けます。
- **プログラムのための設計。** `admetus endpoints --json` が全エンドポイントと出力列を説明するので、
  呼び出し側はこのファイルを読まずに呼び出しを組み立てられます。`admetus doctor` は 1 回の呼び出しで
  「予測できるか、できないなら何を直すか」を答えます。

## 仕組み

エンドポイントごとに個別に学習します。データセット間で化合物も単位もアッセイも共有していないためです。

1. **公開された行をそのままパースする。** SMILES は RDKit に渡すだけで、他に何もしません。塩の
   除去も電荷の中和も重複除去もありません。したがって同じ分子を指す 2 行は 2 行のまま残り、
   分割の両側に入り得ます。どの程度起きるかは後述の「限界」に記載しています。
2. **特徴量化する。** Morgan フィンガープリントと 2D ファーマコフォアシグネチャを連結します。
   ファーマコフォア側は 2 点または 3 点のファーマコフォア特徴の組と各ペアのトポロジカル距離を
   列挙するので、1 ビットが「ドナーとアクセプターが 4〜5 結合離れ、芳香環が両方から 6〜7 結合」
   のような意味を持ちます。
3. **ランダムに 20% をホールドアウトし**（seed 固定）、残りを fold に分ける。fold ごとに
   1 モデルを学習し、各モデルは自分の検証行で early stopping する。アンサンブルはその平均で、
   モデル間のばらつきを予測値の隣に出す。
4. **fold ごとに個別にチューニングして学習する。** 自動チューナーを fold ごとに 1 回実行し、
   その fold の検証行を watchlist として使い、チューナーが選んだ booster をアンサンブルの
   メンバーにする。したがって fold ごとにパラメータが異なり、試行数も fold ごとになる。
5. **1 度だけ測る。** ホールドアウトした 20% を 1 度だけ採点し、エンドポイント固有の指標と
   ブートストラップ区間で報告する。

データには途中で他に何もしません。標準化なし、重複の畳み込みなし、ラベル変換なし、出力の
クランプなし、列の枝刈りなし、確率の校正なし。ラベルは公開されたままの値で、LightGBM には
89,852 列のベクトルをそのまま渡します。これらはいずれも一度ここで実装して削除したもので、
複数は実測で有害でした。各々の代償は `docs/benchmark.md` に記録しています。

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
`admetus evaluate` が報告するのは本プロジェクト自身のランダムホールドアウトです。

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
| `<key>` | 予測値。fold モデル間の平均。分類では LightGBM が出す確率で、**校正はしていません**。回帰では公開されている尺度の上での値 |
| `<key>_ens_sd` | fold アンサンブル間の不一致。log 尺度のエンドポイントでは `<key>_sd_log10` という名前になる。log 値の標準偏差は公開単位での標準偏差ではないため |

さらに `admetus_error` 列が、分子を使えなかったときの理由を持ちます。

`--column-prefix` はすべてに接頭辞を付けるので、自分の列との衝突を構造的に不可能にできます。

以前はここに適用領域フラグと conformal 予測区間がありました。どちらも削除しました。実測すると
フラグは 29 エンドポイント中 16 で情報を持たず（`dili` は水を圏内と判定しました）、区間は回帰 9
エンドポイント中 8 で過剰にカバーし使用に耐えませんでした。どちらも要件になく、安全装置に見えて
そうでない列は、列が無いより悪いからです。

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

以下の数値はすべて、このリポジトリで再現できる実行で測ったものです。推定値はひとつもなく、
空欄を埋めてもいません。

### ホールドアウト（ランダム 20%）

分割は参照実装のものです。`train_test_split(test_size=0.2, random_state=0)` の後、学習側を
3 fold に分けます。ホールドアウトした 5 分の 1 は学習後に 1 度だけ採点し、区間はブートストラップ
2,000 回で求めます。ここに対して選択は一切行いません（ハイパーパラメータも特徴量空間も）。
選択に使った交差検証スコアは各バンドルに「biased」と明記して記録してあり、どこにも引用しません。

| Endpoint | Task | Metric | Held-out | 95% CI | n |
|---|---|---|---|---|---|
| `caco2` | regression | mae | 0.3042 | [0.2673, 0.3426] | 182 |
| `pampa` | classification | auroc | 0.7550 | [0.6867, 0.8244] | 407 |
| `hia` | classification | auroc | 0.9948 | [0.9809, 1.0000] | 116 |
| `pgp` | classification | auroc | 0.9468 | [0.9172, 0.9724] | 244 |
| `bioavailability` | classification | auroc | 0.7549 | [0.6330, 0.8680] | 128 |
| `lipophilicity` | regression | mae | 0.5167 | [0.4853, 0.5478] | 840 |
| `solubility` | regression | mae | 0.7797 | [0.7442, 0.8171] | 1996 |
| `bbb` | classification | auroc | 0.9244 | [0.8900, 0.9551] | 408 |
| `ppbr` | regression | mae | 8.2552 | [7.2838, 9.3094] | 323 |
| `vdss` | regression | spearman | 0.3051 | [0.1762, 0.4290] | 226 |
| `cyp2c9_veith` | classification | auprc | 0.8362 | [0.8108, 0.8606] | 2419 |
| `cyp2d6_veith` | classification | auprc | 0.7587 | [0.7210, 0.7953] | 2626 |
| `cyp3a4_veith` | classification | auprc | 0.8791 | [0.8614, 0.8963] | 2466 |
| `cyp1a2_veith` | classification | auprc | 0.9318 | [0.9207, 0.9426] | 2516 |
| `cyp2c19_veith` | classification | auprc | 0.8758 | [0.8570, 0.8940] | 2533 |
| `cyp2c9_substrate` | classification | auprc | 0.3871 | [0.2440, 0.5763] | 134 |
| `cyp2d6_substrate` | classification | auprc | 0.5282 | [0.3830, 0.6996] | 134 |
| `cyp3a4_substrate` | classification | auroc | 0.6455 | [0.5495, 0.7342] | 134 |
| `half_life` | regression | spearman | 0.4355 | [0.2826, 0.5668] | 134 |
| `clearance_hepatocyte` | regression | spearman | 0.4269 | [0.3136, 0.5283] | 243 |
| `clearance_microsome` | regression | spearman | 0.6278 | [0.5430, 0.6984] | 221 |
| `ld50` | regression | mae | 0.4391 | [0.4197, 0.4590] | 1477 |
| `herg` | classification | auroc | 0.8667 | [0.7965, 0.9270] | 131 |
| `herg_karim` | classification | auroc | 0.9240 | [0.9137, 0.9334] | 2689 |
| `ames` | classification | auroc | 0.9027 | [0.8872, 0.9177] | 1456 |
| `dili` | classification | auroc | 0.8793 | [0.7920, 0.9465] | 95 |
| `carcinogens` | classification | auprc | 0.8164 | [0.5499, 0.9753] | 56 |
| `skin_reaction` | classification | auroc | 0.7778 | [0.6567, 0.8874] | 81 |
| `tox21_nr_ar` | classification | auprc | 0.5257 | [0.4111, 0.6287] | 1452 |

学習は `admetus train --all --time-budget 240`、すなわち fold あたり 240 秒のチューニングで、
エンドポイントあたり 12 分です。1 行の再現は `admetus train --endpoint dili --time-budget 240`、
読み出しは `admetus evaluate` です。

### 公式プロトコル

TDC ADMET ベンチマークグループの 22 エンドポイントを、各データセットが公開している `train_val`
分割で学習し、公開 `test` 分割で採点します。5 シードで、リーダーボードと同じ集計です。
**現在実行中**です。この表は実行のレポートから生成するもので、推定では埋めないため、完了まで
空のままにしてあります。

どちらの表も `python scripts/benchmark_tables.py --models-dir models` で再生成します。バンドルから
読み出すので手打ちしません。2 つの README は同じ数値を持つ必要があり、CI が数値セルを全件比較します。

### 読み方

- 600 行未満のデータセットは記録に `low_confidence` が付き、区間がその理由を示します。
  `carcinogens` は 56 行のテストで 0.8164、95% 区間は 0.5499〜0.9753 です。
- `cyp2c9_substrate` と `cyp2d6_substrate` は 134 行・陽性少数での AUPRC です。点推定ではなく
  区間を読んでください。
- ホールドアウト列は「行」のホールドアウトスコアであり、「分子」のホールドアウトではありません。
  重複除去をしないので、2 回公開された分子は両側に入り得ます。Caco-2 のテスト 182 行のうち
  26 行は、学習行と同じ特徴量ベクトルを持ちます。
- 交差検証スコアは性能ではありません。29 エンドポイントのうち 26 で out-of-fold はモデルを
  実際より良く見せ、中央値で 0.045 の差がありました。

スループット・メモリ・削除した各段の代償といったエンジニアリング面の測定値は
[docs/benchmark.md](docs/benchmark.md) にあります。

## 設定

特徴量空間は全エンドポイントで 1 つです。Morgan 半径 6・2,048 ビットに 2 点および 3 点の
薬理フォアシグネチャを連結した 89,852 列で、選ぶ余地はなく、選ぶためのフラグもありません。

データセットの大きさから決まるものは 2 つだけ残っており、どちらも特徴量空間には触れません。
学習側が支えられる fold 数と、指標を点推定として読めない小ささかどうか（`low_confidence`）です。

| フラグ | 既定 | 変わるもの |
|---|---|---|
| `--folds` | 3 | 学習側の fold 数、つまりアンサンブルのメンバー数 |
| `--time-budget` | 900 | **1 fold** のチューニングの実時間上限（秒） |
| `--seed` | 0 | ホールドアウトの分割とチューナーのサンプラー |
| `--jobs` | 0 | LightGBM のスレッド数。0 は LightGBM に任せる |

## 制限

- **学習済みモデルはまだ公開していません。** ローカルで学習してください。
- **2D のみ。** コンフォマーを生成しないので、立体化学はモデルから見えません。鏡像異性体は同じ
  予測値になります。
- **ホールドアウトの数値は「学習に使っていない行」での成績で、「未知の構造」での成績ではありません。**
  参照実装と同じく、重複を含めて公開されたままの行を分割します。したがって同じ分子が両側に現れ得ます。
  Caco-2・シード 0 では **テスト 182 行のうち 26 行が学習側の行と同一の特徴ベクトル**です。
  参照実装が測っているとおり「適合に使っていない行での成績」として読み、「見たことのない化学構造での
  成績」とは読まないでください。
- **適用領域フラグはありません。** 以前はありましたが、実測で 29 エンドポイント中 16 で情報を
  持たなかったため、安全装置に見える状態で残すより削除しました。入力分子が学習データに似ているか
  を示す出力は無いので、創薬対象から外れた化学構造を扱う場合は呼び出し側で判断してください。
- **予測区間はありません。** conformal 区間がありましたが、回帰 9 エンドポイント中 8 で過剰に
  カバーし使えませんでした。残るのは `<key>_ens_sd` で、これは食い違いの尺度であり誤差範囲では
  ありません。
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
