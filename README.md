# sicoreパッケージ
Selective Inferenceで共通して利用される機能をまとめました．

特定の用途に最適化されていないため，実行速度は遅めです．

## 導入
インストールにはPython3.6以上が必要．依存パッケージは自動的にインストールされます．
```
$ pip install --upgrade setuptools wheel
$ pip install .
```
アンインストール：
```
$ pip uninstall sicore
```


## 機能一覧
`from sicore import *`でインポートされる機能は以下の通りです．

**検定**
- NaiveInferenceNorm：正規分布に従う検定統計量に対するnaiveな検定
- SelectiveInferenceNorm：正規分布に従う検定統計量に対するselectiveな検定
    - Selection Eventの次数は制限なし
    - Parametric SI対応
- SelectiveInferenceNormSE：正規分布に従う検定統計量に対するselectiveな検定
    - 古い実装（互換性保持のため残している）
    - 2次までのSelection Eventにしか対応しないが，SelectiveInferenceNormより高速
    - Parametric SI非対応
- two_sample_test()：naiveな1標本検定
- one_sample_test()：naiveな2標本検定

**切断分布**  
全て複数の切断区間に対応し，mpmathを用いた任意精度の計算
- tn_cdf()：切断正規分布
- tt_cdf()：切断t分布
- tc2_cdf()：切断カイ2乗分布
- tf_cdf()：切断F分布

**検定の評価**
- false_positive_rate()
- false_negative_rate()
- true_negative_rate()
- true_positive_rate()
- type1_error_rate()：false_positive_rate()のエイリアス
- type2_error_rate()：false_negative_rate()のエイリアス
- power()：true_positive_rate()のエイリアス

**図の描画**
- pvalues_hist()：p値のヒストグラムを描画
- pvalues_qqplot()：p値の一様Q-Qプロットを描画
- FprFigure：FPRの実験図を描画
- PowerFigure：検出力の実験図を描画

**区間**
- intervals.intersection()：2つの区間の積を計算
- intervals.intersection_all()：複数区間の積を計算
- intervals.union_all()：複数区間の和を計算
- intervals.not_()：実数上で区間の補集合を計算
- intervals.poly_lt_zero()：多項式の0以下となる区間を計算

**その他の便利な機能**
- OneVec：特定の場所が1，それ以外が0のベクトルを生成


## その他
テストの実行：
```
$ pytest tests/
```