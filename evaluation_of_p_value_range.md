# 真の $p$ 値を含み幅が $0$ へと収束する閉区間の縮小列の構成法

## 記法の導入
- $t$ を検定統計量の実現値とする.
- $F$ を帰無仮説のもとで検定統計量の従う分布とする.
- $\mathbb{R}$ の部分集合族 $\mathcal{R}$ を次のように定める.

$$
    \mathcal{R} = \lbrace R\subset \mathbb{R}\mid t\in R, \text{$R$ is a union set of closed intervals.}\rbrace
$$
- $g$ を $g\colon \mathcal{R}\ni R\mapsto g(R)\in[0,1]$ なる写像として, $g(R)$ は
- 分布 $F$ の $R$ による切断分布の累積分布関数 $F^R$ の引数として$t$を与えたときの値 $F^R(t)$ とする.
- $p$ を $p\colon \mathcal{R}\ni R\mapsto p(R)\in[0,1]$ なる写像として, $p(R)$ は対象とする
  検定方式に応じて次のように定めるとする.

$$
    p(R)= \begin{cases}
          g(R)                    & \text{(left)}   \\
             1-g(R)                  & \text{(right)}  \\
          2\min\{g(R), 1-g(R)\} & \text{(double)}
       \end{cases}
$$
- $i$ を探索の回数を表す添字とする.
- $\varepsilon$ を探索値のステップ幅を表すハイパーパラメータとする.
- $z_i$を $i$ 回目の探索で探索する実数値とする.
- $R_i$を $i$ 回目までの探索で得られた切断領域とする.
- $S_i$を $i$ 回目までの探索で探索済みの領域とする.
- $U_i$を $i$ 回目までの探索で未探索の領域とする.
- $[s_i, e_i]$ を $S_i$ に含まれる $t$ を含む閉区間のうち幅が最大であるものとする. つまり以下が成立する.

$$
    s_i = \sup\lbrace U_i\cap [-\infty, t]\rbrace
$$

$$
    e_i = \inf\lbrace U_i\cap [t, \infty]\rbrace 
$$

## パラメトリックサーチの順序に対する仮定と成立事項

- 1回目の探索では $z_1=t$ の探索からスタートすることとする.
- 任意の $i$ に対して $[s_{i+1}, e_{i+1}]\supsetneq [s_i, e_i]$
  となるように探索を行う. つまり $z_{i+1}=s_i-\varepsilon$ または $z_{i+1}=e_i+\varepsilon$ とする.
- 任意の $i$ に対して, 定義から明らかに $S_i\cap U_i=\emptyset, S_i\cup U_i=\mathbb{R}, R_i\subset S_i$ が成立する.
- 探索によって, 実数値に対してそれを含むような幅を持つ閉区間
  が得られることから, 高々可算回の探索で全実数域の探索が
  完了するため, 集合族 $\lbrace S_i\rbrace_{i\in\mathbb{N}}$, $\lbrace U_i\rbrace_{i\in\mathbb{N}}$
  が $S_\infty =\mathbb{R}$, $U_\infty =\emptyset$ となるように構成できる.
- 集合族 $\lbrace R_i\rbrace_{i\in\mathbb{N}}$ は $\mathcal{R}$ の部分集合であり, $S_\infty=\mathbb{R}$
  から明らかに $R_\infty$ が真の切断領域となる.
- 真の $p$ 値は明らかに $p(R_\infty)$ となる.
- 任意の $i$ に対して次が成立する.

$$
    R_i \subset R_{i+1} \tag{1}
$$

$$
    R_{i+1}\setminus R_i \subset \mathbb{R}\setminus (s_i,e_i) \tag{2}
$$ 
- まず式(1)は探索によって得られる切断領域は探索が進むごとに拡大していくことを
  表している. また $i+1$ 回目の探索で新たに切断領域に追加される領域は $i+1$ 回目の探索で
  初めて探索される領域に限られることから, $R_{i+1}\setminus R_i \subset S_{i+1}\setminus S_i$
  が成立して, これと $S_{i+1}\subset \mathbb{R}$ , $(s_i, e_i)\subset S_i$ を合わせることで
  式(2)が得られる.
- $\mathcal{R}$ の部分集合として $\mathbb{R}$ の部分集合族 $\mathcal{R}_i$ を次のように定める.

$$
    \mathcal{R}_i = \lbrace R\in\mathcal{R}\mid R_i\subset R, 
    R\setminus R_i\subset \mathbb{R}\setminus (s_i,e_i)\rbrace
$$
- 式(1), (2)と定義から明らかに集合列 $\lbrace \mathcal{R} _ i \rbrace_{i\in\mathbb{N}}$
  は単調減少列となる. 特に定義から $\mathcal{R}_ \infty=\lbrace R_\infty \rbrace$
  が成立するため, 任意の $i$ に対して $R_\infty\in\mathcal{R}_ i$ となる.

## 目的の閉区間の列の構成

先の議論から任意の$i$に対して $R_\infty\in\mathcal{R}_i$ であるため

$$
    \inf_{R\in\mathcal{R}_i}g(R)\leq g(R_\infty) \leq \sup_{R\in\mathcal{R}_i}g(R) \tag{3}
$$

が成立する. ここで両辺はそれぞれ, 写像 $g$ と集合族 $\mathcal{R}_i$ の定義と $s_i\leq t\leq e_i$ であることから
累積分布関数の定義に注意すれば次のようにして計算することができる.

$$
    \inf_{R\in\mathcal{R}_i}g(R) = F^{R_i\cup [e_i,\infty]}(t)
$$

$$
    \sup_{R\in\mathcal{R}_i}g(R) = F^{R_i\cup [-\infty, s_i]}(t)
$$

また2つの実数列
$\lbrace \inf_{R\in\mathcal{R} _ i} g(R) \rbrace _ {i\in\mathbb{N}}$, $\lbrace \sup_{R\in\mathcal{R} _ i} g(R)\rbrace _ {i\in\mathbb{N}}$
はそれぞれ, 上限と下限の定義と集合列 $\lbrace\mathcal{R}_ i\rbrace_{i\in\mathbb{N}}$ の単調減少性
とから, 単調増加列と単調減少列となる. さらに $\mathcal{R}_ \infty=\lbrace R_\infty\rbrace$ であることから
これらの実数列は共に $g(R_\infty)$ へと収束する.

それぞれの検定方式に対して真の $p$ 値を $p^{\mathrm{left}},p^{\mathrm{right}},p^{\mathrm{double}}$ と
表すこととすれば, 写像 $p$ の定義と上に示した $g(R^\infty)$ に関する不等式とから, 任意の $i$ に対して

$$
\begin{align*}
    p^{\mathrm{left}} & = g(R_\infty)                                                                                    \\
                          & \in [\inf_{R\in\mathcal{R}_i}g(R), \sup_{R\in\mathcal{R}_ i}g(R)] \\&=I_i^{\mathrm{left}}        \\
      p^{\mathrm{right}}  & =1- g(R_\infty)                                                                                  \\
                          & \in [1-\sup_{R\in\mathcal{R}_i}g(R), 1-\inf_{R\in\mathcal{R}_ i}g(R)]   \\&=I_i^{\mathrm{right}} \\
      p^{\mathrm{double}} & = 2\min\{g(R_\infty), 1-g(R_\infty)\}                                                          \\
                          & \in [2\min\{\inf_{R\in\mathcal{R}_ i}g(R), 1-\sup_{R\in\mathcal{R}_ i}g(R)\},
            2\min\{\sup_{R\in\mathcal{R}_ i}g(R), 1-\inf_{R\in\mathcal{R}_ i}g(R)\}]\\&=I_i^{\mathrm{double}}
\end{align*}
$$

が成立する. ここで用いた記号を用いることで, それぞれの検定方式に対して真の $p$ 値を含む閉区間の列
$\lbrace I_i^\mathrm{right}\rbrace_{i\in\mathbb{N}}$,
$\lbrace I_i^\mathrm{left}\rbrace_{i\in\mathbb{N}}$,
$\lbrace I_i^\mathrm{double}\rbrace_{i\in\mathbb{N}}$
を構成することができる. これら3つの閉区間の列はいずれも, 2つの実数列
$\lbrace\inf_{R\in\mathcal{R}_ i}g(R)\rbrace_{i\in\mathbb{N}}$, $\lbrace\sup_{R\in\mathcal{R}_ i}g(R)\rbrace_{i\in\mathbb{N}}$
の単調性に注意すれば縮小列となり, さらにこれらの実数列の収束値が等しいことに注意すれば幅は $0$ へと収束する.

以上から目的の閉区間の列が構成された. また実用上は, これらの閉区間の列の任意の要素は先に示した通り
$R_i$ と $U_i$ から容易に計算可能である.
