---
title: 'Boundary-based-spatial-segregation-of-dissimilarity'
date: 2020-11-14
permalink: /posts/2020/11/Boundary-based-spatial-segregation-of-dissimilarity/
tags:
  - segregation study
  - Chicago
  - dissimilarity
---


------

这篇blog主要目的是记录关于dissimilarity的计算中引入spatial component的工作。

 Let region $R$ 被分割为$n$ 个不重叠的区域,我们用$i$或者$j$索引$i,j\in1,2,\ldots,n$，and there are $M$ racial groups in $R$ and indexed by $m$, $m\in1,2,\ldots,M$. Let $\tau_{i,t}$ and $\tau_{i,m,t}$ be population size in unit $i$ at time $t$ and the population size of group $m$ in unit $i$ at time $t$, respectively. The total population at time $t$ denote as $T_t$: 
$$
T_t=\sum_{i=1}^{n}\tau_{i,t}	\tag{1}
$$
The proportion in group $m$ of total population in $R$ is defined as $\pi_{m,t}$ 
$$
\pi_{m,t}=\ \frac{\sum_{i=1}^n\tau_{i,m,t}\ }{T_t}	\tag{2}
$$
