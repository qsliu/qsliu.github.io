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

## 2. single value inference

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

### 2.1 “systematic”

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

### 2.2 "bootstrap"           

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

### 2.3 "evenness"

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

![image-20201113103449222](/images/blog/2020-11-11/image-20201113103217335.png)

### 2.4 "permutation"            

: randomly allocates the units over space keeping the original values.

```python
###############
# PERMUTATION #
###############
if (null_approach == "permutation"):
    
    for i in np.array(range(iterations_under_null)):
        data = data.assign(geometry=data['geometry'][list(np.random.choice(data.shape[0], data.shape[0],replace=False))].reset_index()['geometry'])
        df_aux = data
        Estimates_Stars[i] = seg_class._function(df_aux, 'group_pop_var', 'total_pop_var', **kwargs)[0]

                
```



### 2.5 "systematic_permutation" 

: assumes absence of systematic segregation and randomly allocates the units over space. 

就是先进行 systematric分配，在permutation.

```python
if (null_approach == "systematic_permutation"):
    
    if ('multigroup' not in str(type(seg_class))):

        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')

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

        with tqdm(total=iterations_under_null) as pbar:
            for i in np.array(range(iterations_under_null)):
                data_aux = {
                    'simul_group': sim0[i].tolist(),
                    'simul_tot': (sim0[i] + sim1[i]).tolist()
                }
                df_aux = pd.DataFrame.from_dict(data_aux)
                df_aux = gpd.GeoDataFrame(df_aux)
                df_aux['geometry'] = data['geometry']
                df_aux = df_aux.assign(geometry=df_aux['geometry'][list(
                    np.random.choice(
                        df_aux.shape[0], df_aux.shape[0],
                        replace=False))].reset_index()['geometry'])
                Estimates_Stars[i] = seg_class._function(
                    df_aux, 'simul_group', 'simul_tot', **kwargs)[0]

                pbar.set_description(
                    'Processed {} iterations out of {}'.format(
                        i + 1, iterations_under_null))
                pbar.update(1)
                
    if ('multigroup' in str(type(seg_class))):
        raise ValueError('Not implemented for MultiGroup indexes.')
```



### 2.6 "even_permutation"       

: assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.

就是先进行 evenness 分配，在permutation.

```python
if (null_approach == "even_permutation"):
    
    if ('multigroup' not in str(type(seg_class))):

        if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
            raise TypeError('data is not a GeoDataFrame, therefore, this null approach does not apply.')

        p_null = data['group_pop_var'].sum() / data['total_pop_var'].sum()

        with tqdm(total=iterations_under_null) as pbar:
            for i in np.array(range(iterations_under_null)):
                sim = np.random.binomial(n=np.array(
                    [data['total_pop_var'].tolist()]),
                                         p=p_null)
                data_aux = {
                    'simul_group': sim[0],
                    'simul_tot': data['total_pop_var'].tolist()
                }
                df_aux = pd.DataFrame.from_dict(data_aux)
                df_aux = gpd.GeoDataFrame(df_aux)
                df_aux['geometry'] = data['geometry']
                df_aux = df_aux.assign(geometry=df_aux['geometry'][list(
                    np.random.choice(
                        df_aux.shape[0], df_aux.shape[0],
                        replace=False))].reset_index()['geometry'])
                Estimates_Stars[i] = seg_class._function(
                    df_aux, 'simul_group', 'simul_tot', **kwargs)[0]
                pbar.set_description(
                    'Processed {} iterations out of {}'.format(
                        i + 1, iterations_under_null))
                pbar.update(1)
                
    if ('multigroup' in str(type(seg_class))):
        raise ValueError('Not implemented for MultiGroup indexes.')
```



## 3 two value inference

* "**random_label**"               : random label the data in each iteration
  
* "**counterfactual_composition**" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.
  
* "**counterfactual_share**" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.
  
* "**counterfactual_dual_composition**" : applies the "counterfactual_composition" for both minority and complementary groups.

### 3.1 random label



```python
################
# RANDOM LABEL #
################
if (null_approach == "random_label"):
    data_1['grouping_variable'] = 'Group_1'
    data_2['grouping_variable'] = 'Group_2'
    
    if ('multigroup' not in str(type(seg_class_1))):
        
        # This step is just to make sure the each frequecy column is integer for the approaches and from the same type in order to be able to stack them
        data_1['group_pop_var'] = round(data_1['group_pop_var']).astype(int)
        data_1['total_pop_var'] = round(data_1['total_pop_var']).astype(int)

        data_2['group_pop_var'] = round(data_2['group_pop_var']).astype(int)
        data_2['total_pop_var'] = round(data_2['total_pop_var']).astype(int)

        stacked_data = pd.concat([data_1, data_2], ignore_index=True)

        with tqdm(total=iterations_under_null) as pbar:
            for i in np.array(range(iterations_under_null)):

                stacked_data['grouping_variable'] = np.random.permutation(stacked_data['grouping_variable'])

                stacked_data_1 = stacked_data.loc[stacked_data['grouping_variable'] == 'Group_1']
                stacked_data_2 = stacked_data.loc[stacked_data['grouping_variable'] == 'Group_2']

                simulations_1 = seg_class_1._function(stacked_data_1,'group_pop_var','total_pop_var',**kwargs)[0]
                simulations_2 = seg_class_2._function(stacked_data_2,'group_pop_var','total_pop_var',**kwargs)[0]

                est_sim[i] = simulations_1 - simulations_2
                pbar.set_description('Processed {} iterations out of {}'.format(i + 1, iterations_under_null))
                pbar.update(1)
```

主要函数 `np.random.permutation`

`numpy.random.``permutation`(*x*)

Randomly permute a sequence, or return a permuted range.

If *x* is a multi-dimensional array, it is only shuffled along its first index.

```python
np.random.permutation([1, 4, 9, 12, 15])
#array([15,  1,  9,  4, 12]) # random
```



![image-20201113103621207](/images/blog/2020-11-11/image-20201113103621207.png)



### 3.2 "**counterfactual**" 

: randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.

```python
##############################
# COUNTERFACTUAL COMPOSITION #
##############################
if (null_approach in ['counterfactual_composition', 'counterfactual_share','counterfactual_dual_composition']):
    
    if ('multigroup' in str(type(seg_class_1))):
        raise ValueError('Not implemented for MultiGroup indexes.')

    internal_arg = null_approach[15:]  # Remove 'counterfactual_' from the beginning of the string

    counterfac_df1, counterfac_df2 = _generate_counterfactual(
        data_1,
        data_2,
        'group_pop_var',
        'total_pop_var',
        counterfactual_approach=internal_arg)

    if (null_approach in [
            'counterfactual_share', 'counterfactual_dual_composition'
    ]):
        data_1['total_pop_var'] = counterfac_df1[
            'counterfactual_total_pop']
        data_2['total_pop_var'] = counterfac_df2[
            'counterfactual_total_pop']
    with tqdm(total=iterations_under_null) as pbar:
        for i in np.array(range(iterations_under_null)):

            data_1['fair_coin'] = np.random.uniform(size=len(data_1))
            data_1['test_group_pop_var'] = np.where(
                data_1['fair_coin'] > 0.5, data_1['group_pop_var'],
                counterfac_df1['counterfactual_group_pop'])

            # Dropping to avoid confusion in the internal function
            data_1_test = data_1.drop(['group_pop_var'], axis=1)

            simulations_1 = seg_class_1._function(data_1_test,
                                                  'test_group_pop_var',
                                                  'total_pop_var',
                                                  **kwargs)[0]

            # Dropping to avoid confusion in the next iteration
            data_1 = data_1.drop(['fair_coin', 'test_group_pop_var'],
                                 axis=1)

            data_2['fair_coin'] = np.random.uniform(size=len(data_2))
            data_2['test_group_pop_var'] = np.where(
                data_2['fair_coin'] > 0.5, data_2['group_pop_var'],
                counterfac_df2['counterfactual_group_pop'])

            # Dropping to avoid confusion in the internal function
            data_2_test = data_2.drop(['group_pop_var'], axis=1)

            simulations_2 = seg_class_2._function(data_2_test,
                                                  'test_group_pop_var',
                                                  'total_pop_var',
                                                  **kwargs)[0]

            # Dropping to avoid confusion in the next iteration
            data_2 = data_2.drop(['fair_coin', 'test_group_pop_var'],
                                 axis=1)

            est_sim[i] = simulations_1 - simulations_2

            pbar.set_description(
                'Processed {} iterations out of {}'.format(
                    i + 1, iterations_under_null))
            pbar.update(1)
```

#### 3.2.1 composition

```python
def _generate_counterfactual(
    data1, data2, group_pop_var, total_pop_var, counterfactual_approach="composition"
):
    """Generate a counterfactual variables.

    Given two contexts, generate counterfactual distributions for a variable of
    interest by simulating the variable of one context into the spatial
    structure of the other.

    Parameters
    ----------
    data1 : pd.DataFrame or gpd.DataFrame
        Pandas or Geopandas dataframe holding data for context 1

    data2 : pd.DataFrame or gpd.DataFrame
        Pandas or Geopandas dataframe holding data for context 2

    group_pop_var : str
        The name of variable in both data that contains the population size of the group of interest

    total_pop_var : str
        The name of variable in both data that contains the total population of the unit

    approach : str, ["composition", "share", "dual_composition"]
        Which approach to use for generating the counterfactual.
        Options include "composition", "share", or "dual_composition"

    Returns
    -------
    two DataFrames
        df1 and df2  with appended columns 'counterfactual_group_pop', 'counterfactual_total_pop', 'group_composition' and 'counterfactual_composition'

    """

    df1 = data1.copy()
    df2 = data2.copy()

    if counterfactual_approach == "composition":

        df1["group_composition"] = np.where(
            df1[total_pop_var] == 0, 0, df1[group_pop_var] / df1[total_pop_var]
        )
        df2["group_composition"] = np.where(
            df2[total_pop_var] == 0, 0, df2[group_pop_var] / df2[total_pop_var]
        )

        df1["counterfactual_group_pop"] = (
            df1["group_composition"]
            .rank(pct=True)
            .apply(df2["group_composition"].quantile)
            * df1[total_pop_var]
        )
        df2["counterfactual_group_pop"] = (
            df2["group_composition"]
            .rank(pct=True)
            .apply(df1["group_composition"].quantile)
            * df2[total_pop_var]
        )

        df1["counterfactual_total_pop"] = df1[total_pop_var]
        df2["counterfactual_total_pop"] = df2[total_pop_var]
 
```



有几个重要函数

pandas.DataFrame.quantile[¶](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html#pandas-dataframe-quantile)

- `DataFrame.quantile`(*q=0.5*, *axis=0*, *numeric_only=True*, *interpolation='linear'*)[[source\]](https://github.com/pandas-dev/pandas/blob/v1.1.4/pandas/core/frame.py#L8981-L9075)

  Return values at the given quantile over requested axis.

举个例子

```python
df = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9,10]))
df.quantile(0.1)
# 1.9 = 1+(10-1)*0.1
df.quantile(0.5)
#5.5 = 1+(10-1)*0.5
```

```python
df = pd.DataFrame(np.array([1,8,8.1,8.2,8.3,8.4,8.5,8.6,9,10]))
df.quantile(0.1)
# 7.3 这里不知道怎么算的。
df.quantile(0.5)
# 8.35
```



```python
df1 = pd.DataFrame(np.array([[5,10],[4,10],[3,10],[2,10],[4,10],[1,10]]),columns=['a','b'])
df2 = pd.DataFrame(np.array([[1,10],[4,10],[2,10],[3,10],[4,10],[5,10]]),columns=['a','b'])
df1["group_composition"] = np.where(
    df1["b"] == 0, 0, df1["a"] / df1["b"]
)
df2["group_composition"] = np.where(
    df2["b"] == 0, 0, df2["a"] / df2["b"]
)
df1["counterfactual_group_pop"] = (
    df1["group_composition"]
    .rank(pct=True)
    .apply(df2["group_composition"].quantile)
    * df1["b"]
)
df2["counterfactual_group_pop"] = (
    df2["group_composition"]
    .rank(pct=True)
    .apply(df1["group_composition"].quantile)
    * df2["b"]
)
```





```python
 if counterfactual_approach == "share":

        df1["compl_pop_var"] = df1[total_pop_var] - df1[group_pop_var]
        df2["compl_pop_var"] = df2[total_pop_var] - df2[group_pop_var]

        df1["share"] = np.where(
            df1[total_pop_var] == 0, 0, df1[group_pop_var] / df1[group_pop_var].sum()
        )
        df2["share"] = np.where(
            df2[total_pop_var] == 0, 0, df2[group_pop_var] / df2[group_pop_var].sum()
        )

        df1["compl_share"] = np.where(
            df1["compl_pop_var"] == 0,
            0,
            df1["compl_pop_var"] / df1["compl_pop_var"].sum(),
        )
        df2["compl_share"] = np.where(
            df2["compl_pop_var"] == 0,
            0,
            df2["compl_pop_var"] / df2["compl_pop_var"].sum(),
        )

        # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
        # CT stands for Correction Term
        CT1_2_group = df1["share"].rank(pct=True).apply(df2["share"].quantile).sum()
        CT2_1_group = df2["share"].rank(pct=True).apply(df1["share"].quantile).sum()

        df1["counterfactual_group_pop"] = (
            df1["share"].rank(pct=True).apply(df2["share"].quantile)
            / CT1_2_group
            * df1[group_pop_var].sum()
        )
        df2["counterfactual_group_pop"] = (
            df2["share"].rank(pct=True).apply(df1["share"].quantile)
            / CT2_1_group
            * df2[group_pop_var].sum()
        )

        # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
        # CT stands for Correction Term
        CT1_2_compl = (
            df1["compl_share"].rank(pct=True).apply(df2["compl_share"].quantile).sum()
        )
        CT2_1_compl = (
            df2["compl_share"].rank(pct=True).apply(df1["compl_share"].quantile).sum()
        )

        df1["counterfactual_compl_pop"] = (
            df1["compl_share"].rank(pct=True).apply(df2["compl_share"].quantile)
            / CT1_2_compl
            * df1["compl_pop_var"].sum()
        )
        df2["counterfactual_compl_pop"] = (
            df2["compl_share"].rank(pct=True).apply(df1["compl_share"].quantile)
            / CT2_1_compl
            * df2["compl_pop_var"].sum()
        )

        df1["counterfactual_total_pop"] = (
            df1["counterfactual_group_pop"] + df1["counterfactual_compl_pop"]
        )
        df2["counterfactual_total_pop"] = (
            df2["counterfactual_group_pop"] + df2["counterfactual_compl_pop"]
        )
```







```python

    if counterfactual_approach == "dual_composition":

        df1["group_composition"] = np.where(
            df1[total_pop_var] == 0, 0, df1[group_pop_var] / df1[total_pop_var]
        )
        df2["group_composition"] = np.where(
            df2[total_pop_var] == 0, 0, df2[group_pop_var] / df2[total_pop_var]
        )

        df1["compl_pop_var"] = df1[total_pop_var] - df1[group_pop_var]
        df2["compl_pop_var"] = df2[total_pop_var] - df2[group_pop_var]

        df1["compl_composition"] = np.where(
            df1[total_pop_var] == 0, 0, df1["compl_pop_var"] / df1[total_pop_var]
        )
        df2["compl_composition"] = np.where(
            df2[total_pop_var] == 0, 0, df2["compl_pop_var"] / df2[total_pop_var]
        )

        df1["counterfactual_group_pop"] = (
            df1["group_composition"]
            .rank(pct=True)
            .apply(df2["group_composition"].quantile)
            * df1[total_pop_var]
        )
        df2["counterfactual_group_pop"] = (
            df2["group_composition"]
            .rank(pct=True)
            .apply(df1["group_composition"].quantile)
            * df2[total_pop_var]
        )

        df1["counterfactual_compl_pop"] = (
            df1["compl_composition"]
            .rank(pct=True)
            .apply(df2["compl_composition"].quantile)
            * df1[total_pop_var]
        )
        df2["counterfactual_compl_pop"] = (
            df2["compl_composition"]
            .rank(pct=True)
            .apply(df1["compl_composition"].quantile)
            * df2[total_pop_var]
        )

        df1["counterfactual_total_pop"] = (
            df1["counterfactual_group_pop"] + df1["counterfactual_compl_pop"]
        )
        df2["counterfactual_total_pop"] = (
            df2["counterfactual_group_pop"] + df2["counterfactual_compl_pop"]
        )
```



未完。。。

## Reference:

Cortes, R.X., Rey, S., Knaap, E., Wolf, L.J., 2020. An open-source framework for non-spatial and spatial segregation measures: the PySAL segregation module, Journal of Computational Social Science. Springer Singapore. doi:10.1007/s42001-019-00059-3

Rey, S.J., Sastré-Gutiérrez, M.L., 2010. Interregional inequality dynamics in Mexico. Spat. Econ. Anal. 5, 277–298. doi:10.1080/17421772.2010.493955

Yao, J., Wong, D.W.S., Bailey, N., Minton, J., 2018. Spatial Segregation Measures: A Methodological Review. Tijdschr. voor Econ. en Soc. Geogr. 00. doi:10.1111/tesg.12305