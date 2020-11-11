---
title: 'Inference test by simulation'
date: 2020-11-11
permalink: /posts/2020/11/inference-test-by-simulation/
toc:
tags:
  - inference test
  - simulation
  - permutation
---



## 1. Inference

推断在现代地理学的计算中是很重要的一个方面。

比如，在Cortes et al. (2020) 的工作中，就对传统的random labeling approach进行了扩展，可以对（不同时间或同一时间的）不同区域的segregation index 进行统计推断：

> The major contribution of our framework is the ability to perform inference to compare more than one segregation measure. To do so, we extend (Rey and Sastré-Gutiérrez, 2010), who provide an inferential basis for comparisons of regional statistics. Their approach relies on a **random labeling approach**, where in each permutation, each unit in the dataset is assigned randomly to a point in time. However, our approach for comparative segregation stands as more generic and may be applied in any situation where two spatial contexts are compared. 

Cortes et al. (2020) 举了两个例子。

> For example, a user can compare the evolution of a single region between two points in time, two regions in the same point in time, and, also, two regions between two points in time. The first case is a straightforward adaptation of (Rey and Sastré-Gutiérrez, 2010), but the second differs, given the possibility that each region may have entirely different spatial contexts. 

为了解决问题，作者提出，

> To try to provide alternative ways to assess the absence of segregation difference, our framework comprises not only a random data labeling (“random label” approach), but also a labeling process that randomizes observations according to the cumulative distribution function representing the population share for the group of interest in each unit (“counterfactual composition” approach).

 在 Yao et al. (2018) 中也提到，

> Statistical inferences 
>
> As discussed above, spatial segregation measures can be sensitive to the representation of spatial interaction and spatial scale. It is desirable to know the uncertainty associated with the derived indi- ces. This issue was investigated to a certain degree soon after the dissimilarity index became popular (e.g. Cortese et al. 1976). Several studies have attempted to address this issue for non-spatial segregation indices using bootstrap tests (Brulhart & Traeger 2005). However, Lee et al. (2015) found that standard bootstrapping approaches can per- form very badly in the presence of spatial autocorrelation and the new method developed by Lee et al. (2015) performs much better, which has been applied to an centralisation index on the significance of the index difference across space or time (Kavanagh et al. 2016) and can in principle be applied to a wide range of spatial segrega- tion measures.

Yao 同时也指出randomisation approach的缺点。

>However, the randomisation approach to test the significance of segregation indices has its limitations. For instance, when original population counts are randomly reassigned to different areal units, the process implicitly assumes that an areal unit can accommodate any population size, disregarding the area of the unit or other physical constraints. This assumption is unrealistic, and therefore not all randomised distributions are feasible. Effective and conceptually sounded testing approaches are still warranted. In addition, randomness is often regarded as the underlying distribution for no segregation. But if two groups are distributed across areas in the same manner, the definition for no segregation according to D, then higher non-random distributions may still have no segregation. Therefore, the concept of no segregation is still fuzzy.

其实，Cortes et al.的方法弥补了Yao et al.提出的缺点。

在这篇博客中，我们就从Cortes et al.的文章和代码入手，分析一下inference的具体实现。代码中实现了两种类型的inference.

* single value inference
* comparative inference

作者首先给出了 推断 dissimilarity index的显著性的例子

```python
import geopandas as gpd
import segregation
import libpysal
import pandas as pd
import numpy as np

from segregation.inference import SingleValueTest, TwoValueTest
from segregation.aspatial import Dissim
s_map = gpd.read_file("sacramentot2.shp")
D = Dissim(gdf, 'HISP_', 'TOT_POP') #在这里，作者对Dissimilarity的公式理解存疑。因为传统的公式是针对两个族群的,比如White and black
D.statistic
# 0.32184656076566864
infer_D_eve = SingleValueTest(D, iterations_under_null = 1000, null_approach = "evenness", two_tailed = True)
infer_D_eve.plot()
```

![image-20201111141427156](/images/blog/2020-11-11/image-20201111141427156.png)

我们可以明显的看出，实际值比模拟值明显处在差异。

`SingleValueTest`是作者实现的一个Python 类(segregation/inference/inference_wrappers.py).

其他参数很好理解，我们这里只讨论`null_approach参数,作者在代码中这样写道：`

```python
"systematic"             : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
"bootstrap"              : generates bootstrap replications of the units with replacement of the same size of the original data.
"evenness"               : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).

"permutation"            : randomly allocates the units over space keeping the original values.

"systematic_permutation" : assumes absence of systematic segregation and randomly allocates the units over space.
"even_permutation"       : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
```

根据作者的说明，我们很难推断出作者是怎么实现的。但是我们可以看出使用参数permutation, 就是传统的**random labeling approach**。那其他的方法是怎么实现的呢？我们就从代码入手逐个分析。

假设在一个区域$R$ 中，分布着不相互重叠的$n$ 个不同的单元, 我们用$i\in 1,2,...,n$ 来索引。假设区域$R$ 中有$M$个不同的族群，我们用$m \in 1,..,m$ 来索引。每个单元的每个族群人口总数分别是$\tau_{im}$. 那么

每个单元的总人口$T_i = \sum_{m=1}^M \tau_{im}$ where $i=1,...,n$. 

$R$ 中的总人口: $T=\sum_{i=1}^n T_i=\sum_{i=1}^n\sum_{m=1}^M \tau_{im}$

$R$ 中$m$族群的总人口: $T_m = \sum_{i=1}^n \tau_{im}$ where $m \in 1,2,...,M$

### 1.1 “systematic”

假设m族群的人口是均匀分布的，但是受到一些随机因素的影响。那么怎么在体现整体的均匀性的情况下，模拟这些随机因素呢？作者使用从 multinomial distribution中抽样的方法，假设，少数族裔$m$在$n$ 个区域内均匀分布，那
$$
p_1=\frac{\tau_{1m}}{T_m}=\frac{T_1}{T} \\
p_2=\frac{\tau_{2m}}{T_m}=\frac{T_2}{T}\\
\vdots \\
p_n = \frac{\tau_{nm}}{T_m}=\frac{T_n}{T}\\
$$


下面代码中的`p_j` 就是[$p_1,...,p_n$]

```python
##############
# SYSTEMATIC #
##############
if (null_approach == "systematic"):
    data['other_group_pop'] = data['total_pop_var'] - data['group_pop_var']
    p_j = data['total_pop_var'] / data['total_pop_var'].sum() 

    # Group 0: minority group
    p0_i = p_j
    n0 = data['group_pop_var'].sum()
    sim0 = np.random.multinomial(n0, p0_i, size=iterations_under_null)

    # Group 1: complement group
    p1_i = p_j
    n1 = data['other_group_pop'].sum()
    sim1 = np.random.multinomial(n1, p1_i, size=iterations_under_null)
    # ..........
    # Omit some codes
    # ..........
    for i in np.array(range(iterations_under_null)):
        data_aux = {
            'simul_group': sim0[i].tolist(),
            'simul_tot': (sim0[i] + sim1[i]).tolist()
        }
        df_aux = pd.DataFrame.from_dict(data_aux)
    # ..........
    # Omit some codes
    # ..........
```

我们可以看出，关键函数是 **numpy.random.multinomial**

我们从 https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html 找到此函数的文档说明。

`numpy.random.multinomial`(*n*, *pvals*, *size=None*)

Draw samples from a multinomial distribution.

The multinomial distribution is a multivariate generalization of the binomial distribution. Take an experiment with one of `p` possible outcomes. An example of such an experiment is throwing a dice, where the outcome can be 1 through 6. Each sample drawn from the distribution represents *n* such experiments. Its values, `X_i = [X_0, X_1, ..., X_p]`, represent the number of times the outcome was `i`.

为了便于理解，这里有一个投色子的例子。假设我们的色子是质地均匀的，那么色子的6个面朝上的几率是均等的，都是1/6。第二个参数[1/6.,1/6.,1/6.,1/6.,1/6.,1/6.] 就代表了每个面朝上的概率。第一个参数20，表示我们掷了20次色子。返回的结果[4, 1, 7, 5, 2, 1] 就是做完20次实验之后每个面朝上的模拟次数。

```python
np.random.multinomial(20, [1/6.,1/6.,1/6.,1/6.,1/6.,1/6.], size=1)
#array([[4, 1, 7, 5, 2, 1]]) # random
```

我们可以更改 `size` 参数, 来做很多组实验。每组掷20次色子。

```python
np.random.multinomial(20, [1/6.]*6, size=2)
array([[3, 4, 3, 3, 4, 3], # random
       [2, 4, 3, 4, 0, 7]])
```

### 1.2 "bootstrap"           

  : generates bootstrap replications of the units with replacement of the same size of the original data.

Bootstrap就是有放回的随机抽取。主要利用**np.random.choice**

```python
#############
# BOOTSTRAP #
#############
if (null_approach == "bootstrap"):
    
    with tqdm(total=iterations_under_null) as pbar:
        for i in np.array(range(iterations_under_null)):

            sample_index = np.random.choice(data.index, size=len(data), replace=True)
            df_aux = data.iloc[sample_index]
            Estimates_Stars[i] = seg_class._function(df_aux, 'group_pop_var', 'total_pop_var', **kwargs)[0]

```

### 1.3 "evenness"

: assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).

```python
p_null = data['group_pop_var'].sum() / data['total_pop_var'].sum()

for i in np.array(range(iterations_under_null)):
    sim = np.random.binomial(n=np.array([data['total_pop_var'].tolist()]),p=p_null)
    data_aux = {
        'simul_group': sim[0],
        'simul_tot': data['total_pop_var'].tolist()
    }
    df_aux = pd.DataFrame.from_dict(data_aux)

    Estimates_Stars[i] = seg_class._function(df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
```



查阅Python 说明

`numpy.random.binomial`(*n*, *p*, *size=None*)[¶](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html#numpy.random.binomial)

Draw samples from a binomial distribution.

Samples are drawn from a binomial distribution with specified parameters, *n* trials and *p* probability of success where *n* an integer >= 0 and *p* is in the interval [0,1]. (*n* may be input as a float, but it is truncated to an integer in use)

The probability density for the binomial distribution is
$$
P(N) = \begin{pmatrix}n\\N\end{pmatrix} p^N(1-p)^{n-N}
$$
where *n* is the number of trials, *p* is the probability of success, and *N* is the number of successes.

When estimating the **standard error** of a proportion in a population by using a random sample, the normal distribution works well unless the product p\*n <=5, where p = population proportion estimate, and n = number of samples, in which case the binomial distribution is used instead. For example, a sample of 15 people shows 4 who are left handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27\*15 = 4, so the binomial distribution should be used in this case.

当使用随机样本估计人口中比例的**标准误差**时，正态分布效果很好，除非乘积p*n<=5，其中p=人口比例估计，n=样本数，在这种情况下，改用二项分布。例如，一个15人的样本显示4人是左手，11人是右手。那么p=4/15=27%。0.27*15=4，所以在这种情况下应该使用二项分布。

为了便于理解，有以下例子，

```python
np.random.binomial([[10,10,10]],p=0.2)
#array([[1, 3, 1]])
```



## Reference:

Cortes, R.X., Rey, S., Knaap, E., Wolf, L.J., 2020. An open-source framework for non-spatial and spatial segregation measures: the PySAL segregation module, Journal of Computational Social Science. Springer Singapore. doi:10.1007/s42001-019-00059-3

Rey, S.J., Sastré-Gutiérrez, M.L., 2010. Interregional inequality dynamics in Mexico. Spat. Econ. Anal. 5, 277–298. doi:10.1080/17421772.2010.493955

Yao, J., Wong, D.W.S., Bailey, N., Minton, J., 2018. Spatial Segregation Measures: A Methodological Review. Tijdschr. voor Econ. en Soc. Geogr. 00. doi:10.1111/tesg.12305