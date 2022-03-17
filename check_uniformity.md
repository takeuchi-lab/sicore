
# $p$値の一様性チェック

## 視覚的な確認
- ヒストグラム
	- $p$値が[0, 1]上に一様分布することを確認
- 一様Q-Qプロット
	- $U(0, 1)$の分位点と$p$値の経験分布の分位点が$x=y$付近にプロットされるか確認

## 適合度検定
適合度検定 (goodness-of-fit test: GOF test) は2つの分布間の不一致度を測ることで，母集団の確率分布が帰無仮説で提示された分布と異なるかどうか，あるいは2つの母集団の確率分布が異なるかどうかの検定を行う．1標本の適合度検定は，標本$x_1, \dots, x_n$に対して「$H_0$: 母集団の確率分布は分布Fと等しい v.s. $H_1$: 母集団の確率分布は分布Fと異なる」という検定問題を考え，標本の経験分布と分布Fの累積分布関数を比較する．2標本の適合度検定は，標本$x_1, \dots, x_n$と標本$y_1, \dots, y_m$に対して「$H_0$: 2つの母集団の確率分布は等しい v.s. $H_1$: 2つの母集団の確率分布は異なる」という検定問題を考え，2つの標本の経験分布を比較する．

$p$値の一様性チェックは，1標本の適合度検定を用いて帰無仮説に一様分布を仮定する．検定の考え方として帰無仮説が棄却されない場合は判断を保留するため，厳密には「$p$値は一様分布である」と結論づけることはできない点に注意．可能であれば複数の検定方法で実験し比較したほうがいい．

**上限型統計量**：検定統計量は分布間の差の上限
- Kolmogorov–Smirnov検定  (KS検定)
	- 検定統計量の分布が帰無仮説で示された分布に依存しないため非常に便利
		- 正規分布・一様分布・カイ2乗分布など何でも使える
	- 分布の裾よりも中央値付近への感度が高い

**Cramer-von Mises型統計量**：検定統計量は分布間の差の重み付き2乗平均
- Cramer-von Mises検定 (CVM検定)
	- 一様な重みを与える
	- 検定統計量の分布は帰無仮説で示された分布に依存
- Anderson–Darling検定 (AD検定)
	- 分布の裾に大きな重みを与える
		- このため，一様性の検定にはあまり使われない（使うことは可能）
		- 正規分布や裾の重い分布ではKS検定より検出力が高い
	- 検定統計量の分布は帰無仮説で示された分布に依存
		- 一様分布に対する導出はMarsaglia and Marsaglia, "Evaluating the Anderson-Darling Distribution", (2004)を参照

一様性の検定に関するサーベイはBlinov and Lemeshko, "A review of the properties of tests for uniformity" (2014)が詳しい．上記以外のマイナーな手法も網羅されている．

### 一様分布の適合度検定が可能なパッケージ
PythonとRのパッケージの対応状況を以下に示す．RはKS検定 (`ks.test`)をデフォルト搭載．

| | Scipy | statsmodels | scikit-gof | goftest (R) |
----- | :---: | :---: | :---: | :---:
| KS検定 | ○ | × | ○ | × |
| CVM検定 | × | × | ○ | ○ |
| AD検定 | × | △ | ○ | ○ |

- Scipy
	- KS検定 (`scipy.stats.kstest`)
		```
		>>> from scipy.stats import kstest
		>>> kstest(data, 'uniform')
		KstestResult(statistic=0.06586776460315369, pvalue=0.7784507461181711)
		```
	- CVM検定は実装されていない
	- AD検定 (`scipy.stats.anderson`)は一様分布に非対応
	- `scipy.stats.anderson_ksamp`はk標本のAD検定
		- 特定の分布を指定せずに，$k$標本が同じ母集団から抽出されたものかを調べるための手法
- statsmodels
	- KS検定は一様分布に非対応
	- CVM検定は実装されていない
	- AD検定 (`statsmodels.stats.diagnostic.anderson_statistic`)
		- 検定統計量だけしか得られない
		```
		>>> from statsmodels.stats.diagnostic import anderson_statistic
		>>> from scipy.stats import uniform
		>>> anderson_statistic(data, dist=uniform, fit=False)
		0.6773054893310899
		```
- scikit-gof
	- KS検定 (`skgof.ks_test`)
	- CVM検定 (`skgof.cvm_test`)
	- AD検定 (`skgof.ad_test`)
	```
	>>> from skgof import ks_test, cvm_test, ad_test
	>>> from scipy.stats import uniform
	>>> ks_test(data, uniform(0, 1))
	GofResult(statistic=0.06586776460315369, pvalue=0.7532080475207352)
	>>> cvm_test(data, uniform(0, 1))
	GofResult(statistic=0.09590641565930645, pvalue=0.6062656166220206)
	>>> ad_test(data, uniform(0, 1))
	GofResult(statistic=0.6773054893310899, pvalue=0.5776884700201532)
	```
- goftest (Rパッケージ)
	- CVM検定 (`cvm.test`)
	- AD検定 (`ad.test`)
	```
	> require(goftest)
	> cvm.test(data, "punif")

		Cramer-von Mises test of goodness-of-fit
		Null hypothesis: uniform distribution
		Parameters assumed to be fixed

	data:  x
	omega2 = 0.12299, p-value = 0.4849

	> ad.test(data, "punif")

		Anderson-Darling test of goodness-of-fit
		Null hypothesis: uniform distribution
		Parameters assumed to be fixed

	data:  x
	An = 0.70239, p-value = 0.5561
	```
