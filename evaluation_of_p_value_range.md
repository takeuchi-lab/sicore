# 真の $p$ 値を含み幅が $0$ へと収束する閉区間の縮小列の構成法

## 記法の導入
- $t$ を検定統計量の実現値とする.
- $F$ を帰無仮説のもとで検定統計量の従う分布とする.
- $\mathbb{R}$ の部分集合族 $\mathcal{R}$ を次のように定める.

$$
    \mathcal{R} = \lbrace R\subset \mathbb{R}\mid t\in R,\  \text{$R$ is a union set of closed intervals.}\rbrace
$$
- $g$ を $g\colon \mathcal{R}\ni R\mapsto g(R)\in[0,1]$ なる写像として, $g(R)$ は分布 $F$ の $R$ による切断分布の累積分布関数 $F^R$ の引数として $t$ を与えたときの値 $F^R(t)$ とする.
- $p$ を $p\colon \mathcal{R}\ni R\mapsto p(R)\in[0,1]$ なる写像として, $p(R)$ は対象とする検定方式に応じて次のように定めるとする.

$$
    p(R)= \begin{cases}
          g(R)                    & \text{(left)}   \\
             1-g(R)                  & \text{(right)}  \\
          2\min\lbrace g(R), 1-g(R)\rbrace & \text{(double)}
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
- 任意の $i$ に対して $[s_{i+1}, e_{i+1}]\supsetneq [s_i, e_i]$ となるように探索を行う. つまり $z_{i+1}=s_i-\varepsilon$ または $z_{i+1}=e_i+\varepsilon$ とする.
- 任意の $i$ に対して, 定義から明らかに $S_i\cap U_i=\emptyset, S_i\cup U_i=\mathbb{R}, R_i\subset S_i$ が成立する.
- 探索によって, 実数値に対してそれを含むような幅を持つ閉区間が得られることから, 高々可算回の探索で全実数域の探索が完了するため, 集合族 $\lbrace S_i\rbrace_{i\in\mathbb{N}}$, $\lbrace U_i\rbrace_{i\in\mathbb{N}}$ が $S_\infty =\mathbb{R}$, $U_\infty =\emptyset$ となるように構成できる. また集合族 $\lbrace S_i\rbrace_{i\in\mathbb{N}}$ は $\mathcal{R}$ の部分集合である.
- 集合族 $\lbrace R_i\rbrace_{i\in\mathbb{N}}$ は $\mathcal{R}$ の部分集合であり, $S_\infty=\mathbb{R}$ から明らかに $R_\infty$ が真の切断領域となる.
- 真の $p$ 値は明らかに $p(R_\infty)$ となる.
- 任意の $i$ に対して次が成立する.

$$
    R_i \subset R_{i+1} \tag{1}
$$

$$
    R_{i+1}\setminus R_i \subset \mathbb{R}\setminus (s_i,e_i) \tag{2}
$$ 
- まず式(1)は探索によって得られる切断領域は探索が進むごとに拡大していくことを表している. また $i+1$ 回目の探索で新たに切断領域に追加される領域は $i+1$ 回目の探索で初めて探索される領域に限られることから, $R_{i+1}\setminus R_i \subset S_{i+1}\setminus S_i$ が成立して, これと $S_{i+1}\subset \mathbb{R}$ , $(s_i, e_i)\subset S_i$ を合わせることで式(2)が得られる.
- $\mathcal{R}$ の部分集合として $\mathbb{R}$ の部分集合族 $\mathcal{R}_ i$ を次のように定める.

$$
    \mathcal{R}_ i = \lbrace R\in\mathcal{R} \mid R_i\subset R,\ R\setminus R_i\subset 
    \mathbb{R}\setminus (s_i,e_i)\rbrace
$$
- 式(1), (2)と定義に基づけば集合列 $\lbrace \mathcal{R} _ i \rbrace_{i\in\mathbb{N}}$ は単調減少列となる. つまり任意の $i$ に対して, 任意の $R\in\mathcal{R}_ {i+1}$ が $R\in\mathcal{R}_ i$ を満たす. 実際, $R\in\mathcal{R}_ {i+1}$ であることと式(1)とから $R\in \mathcal{R}_ i$がまず成立する. 続いて $R\setminus R_{i+1}\subset \mathbb{R}\setminus (s_{i+1}, e_{i+1})$ が成立していることと式(2)を用いれば次のようにして $R\setminus R_i \subset \mathbb{R}\setminus (s_i, e_i)$ が成立して $R\in\mathcal{R}_ i$ が示される.
  
$$
    \begin{align*}
    R\setminus R_i &= R\cap R_i^c = R\cap R_i^c \cap (R_{i+1}\cup R_{i+1}^c) \\
    &= (R\cap R_{i+1}\setminus R_i)\cup (R\setminus R_{i+1}\cap R_i^c) \\
    &\subset R_{i+1}\setminus R_i \cup R\setminus R_{i+1} \subset \mathbb{R}\setminus (s_i, e_i)
    \end{align*}
$$
- 集合族 $\mathcal{R}_ i$ の定義と集合列 $\lbrace \mathcal{R} _ i \rbrace_{i\in\mathbb{N}}$ の単調減少性から明らかに $\mathcal{R}_ \infty=\lbrace R_\infty \rbrace$ が成立して, 特に任意の $i$ に対して $R_\infty\in\mathcal{R}_ i$ となる.

## 目的の閉区間の列の構成

先の議論から任意の $i$ に対して $R_\infty\in\mathcal{R}_ i$ であるため次が成立する.

$$
    \inf_{R\in\mathcal{R}_ i}g(R)\leq g(R_\infty) \leq \sup_{R\in\mathcal{R}_ i}g(R) \tag{3}
$$

続いて式(3)の両辺をそれぞれ明示的に書き下す. まず写像 $g$ と累積分布関数の定義とから任意の $R\in\mathcal{R}_ i$ に対して $g(R)$ は次のように変形できる.

$$
    \begin{align*}
    g(R) &= F^R(t) = \frac{\int_{R\cap [-\infty, t]}F(x)dx}{\int_R F(x)dx} \\
    &= \frac{\int_{R\cap [-\infty, t]}F(x)dx}{\int_{R\cap [-\infty, t]}F(x)dx + \int_{R\cap [t, \infty]}F(x)dx} \\
    &= \frac{1}{1 + \left( \int_{R\cap [t, \infty]}F(x)dx / \int_{R\cap [-\infty, t]}F(x)dx \right)} 
    \end{align*}
$$

この結果と恒等的に $F(X)\geq 0$ が成立することとから $g(R)$の下限と上限は領域の包含によって評価可能となる. 実際 $R^\prime \in \mathcal{R}_ i$ で任意の $R\in \mathcal{R}_ i$ に対して次に示す式(4)を満たすようなものが存在すれば $\inf_{R\in\mathcal{R}_ i}g(R)=g(R^\prime)$ が成立して, 式(5)を満たすようなものが存在すれば $\sup_{R\in\mathcal{R}_ i}g(R)=g(R^\prime)$ が成立する.

$$
    R^\prime \cap [-\infty, t] \subset R\cap [-\infty, t],\ R^\prime \cap [t, \infty] \supset R\cap [t, \infty] \tag{4}
$$

$$
    R^\prime \cap [-\infty, t] \supset R\cap [-\infty, t],\ R^\prime \cap [t, \infty] \subset R\cap [t, \infty] \tag{5}
$$

ここで $R^\prime$ を $R_i\cup [e_i, \infty]$, $R_i\cup [-\infty, s_i]$ のいずれかとすれば, これらはいずれも $\mathcal{R}_ i$ の元であり, それぞれが式(4), 式(5)を満たす. 実際, 集合族 $\mathcal{R}_ i$ の定義に注意すればその元であることは明らかであり, それぞれが式(4), 式(5)を満たすことは $R\in\mathcal{R}_ i$ であることを用いて次のように示す.

まず $R_i\cup [e_i, \infty]$ については次のような式変形から示される.

$$
    \begin{align*}
    (R_i \cup [e_i, \infty]) \cap [-\infty, t] = R_i\cap [-\infty, t] \subset R\cap [-\infty, t]
    \end{align*}
$$

$$
    \begin{align*}
    R \cap [t, \infty] &= (R_i\cup R\setminus R_i) \cap [t, \infty] \\
                       &= (R_i\cap [t, \infty]) \cup (R\setminus R_i \cap [t, \infty]) \\
                       &\subset (R_i\cap [t, \infty]) \cup (\mathbb{R}\setminus (s_i, e_i) \cap [t,\infty]) \\
                       &= (R_i\cap [t, \infty]) \cup ([e_i, \infty] \cap [t, \infty]) \\
                       &= (R_i\cup [e_i, \infty]) \cap [t, \infty]
    \end{align*}
$$

続いて $R_i\cup [-\infty, s_i]$ についても次のような式変形から示される.

$$
    \begin{align*}
    (R_i \cup [-\infty, s_i]) \cap [t, \infty] = R_i\cap [t, \infty] \subset R\cap [t, \infty]
    \end{align*}
$$

$$
    \begin{align*}
    R \cap [-\infty, t] &= (R_i\cup R\setminus R_i) \cap [-\infty, t] \\
                       &= (R_i\cap [-\infty, t]) \cup (R\setminus R_i \cap [-\infty, t]) \\
                       &\subset (R_i\cap [-\infty, t]) \cup (\mathbb{R}\setminus (s_i, e_i) \cap [-\infty, t]) \\
                       &= (R_i\cap [-\infty, t]) \cup ([-\infty, s_i] \cap [-\infty, t]) \\
                       &= (R_i\cup [-\infty, s_i]) \cap [-\infty, t]
    \end{align*}
$$

以上の結果から最終的には次のようにして式(3)の両辺を明示的に書き下すことができる.

$$
    \inf_{R\in\mathcal{R}_ i}g(R) = F^{R_i\cup [e_i,\infty]}(t) \tag{6}
$$

$$
    \sup_{R\in\mathcal{R}_ i}g(R) = F^{R_i\cup [-\infty, s_i]}(t) \tag{7}
$$

また2つの実数列 $\lbrace \inf_{R\in\mathcal{R} _ i} g(R) \rbrace _ {i\in\mathbb{N}}$, $\lbrace \sup_{R\in\mathcal{R} _ i} g(R)\rbrace _ {i\in\mathbb{N}}$ はそれぞれ, 下限と上限の定義と集合列 $\lbrace\mathcal{R}_ i\rbrace_{i\in\mathbb{N}}$ の単調減少性とから, 単調増加列と単調減少列となる. したがって $g$ の有界性と合わせて, これらの実数列は共に収束する. ここで $\mathcal{R}_ \infty=\lbrace R_\infty\rbrace$ であることと, ふたたび集合列 $\lbrace\mathcal{R}_ i\rbrace_{i\in\mathbb{N}}$ の単調減少性とから, これらの実数列の極限は共に $g(R_\infty)$ となる.

それぞれの検定方式に対して真の $p$ 値を $p^{\mathrm{left}},p^{\mathrm{right}},p^{\mathrm{double}}$ と表すこととすれば, 写像 $p$ の定義と上に示した $g(R^\infty)$ に関する不等式とから, 任意の $i$ に対して

$$
\begin{align*}
    p^{\mathrm{left}} & = g(R_\infty)                                                                                    \\
                          & \in \left[ \inf_{R\in\mathcal{R}_ i}g(R), \sup_{R\in\mathcal{R}_ i}g(R) \right] \\
                          & =I_i^{\mathrm{left}}        \\
      p^{\mathrm{right}}  & =1- g(R_\infty)                                                                                  \\
                          & \in \left[ 1-\sup_{R\in\mathcal{R}_ i}g(R), 1-\inf_{R\in\mathcal{R}_ i}g(R) \right]   \\ 
                          & =I_i^{\mathrm{right}} \\
      p^{\mathrm{double}} & = 2\min\{g(R_\infty), 1-g(R_\infty)\}                                                          \\
                          & \in \left[ 2\min\lbrace\inf_{R\in\mathcal{R}_ i}g(R), 1-\sup_{R\in\mathcal{R}_ i}g(R)\rbrace,
                            2\min\lbrace\sup_{R\in\mathcal{R}_ i}g(R), 1-\inf_{R\in\mathcal{R}_ i}g(R)\rbrace \right] \\
                          & =I_i^{\mathrm{double}}
\end{align*}
$$

が成立する. ここで用いた記号を用いることで, それぞれの検定方式に対して真の $p$ 値を含む閉区間の列 $\lbrace I_i^\mathrm{right}\rbrace_{i\in\mathbb{N}}$, $\lbrace I_i^\mathrm{left}\rbrace_{i\in\mathbb{N}}$, $\lbrace I_i^\mathrm{double}\rbrace_{i\in\mathbb{N}}$ を構成することができる. これら3つの閉区間の列はいずれも, 2つの実数列 $\lbrace\inf_{R\in\mathcal{R}_ i}g(R)\rbrace_{i\in\mathbb{N}}$, $\lbrace\sup_{R\in\mathcal{R}_ i}g(R)\rbrace_{i\in\mathbb{N}}$ の単調性に注意すれば縮小列となり, さらにこれらの実数列の収束値が等しいことに注意すれば幅は $0$ へと収束する.

以上から目的の閉区間の列が構成された. またこれらの閉区間の列の任意の要素は式(6), (7)に基づくことで $R_i$ と $U_i$ から容易に計算可能である.
