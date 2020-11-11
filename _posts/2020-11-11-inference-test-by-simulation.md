---
title: 'Inference test by simulation'
date: 2020-11-11
permalink: /posts/2020/11/inference-test-by-simulation/
tags:
  - inference test
  - simulation
  - permutation
---


------



## Inference

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

在这篇博客中，我们就从Cortes et al.的文章和代码入手，分析一下inference的具体实现。

![image-20201111134434926](images\image-20201111134434926.png)



## Reference:

Cortes, R.X., Rey, S., Knaap, E., Wolf, L.J., 2020. An open-source framework for non-spatial and spatial segregation measures: the PySAL segregation module, Journal of Computational Social Science. Springer Singapore. doi:10.1007/s42001-019-00059-3

Rey, S.J., Sastré-Gutiérrez, M.L., 2010. Interregional inequality dynamics in Mexico. Spat. Econ. Anal. 5, 277–298. doi:10.1080/17421772.2010.493955

Yao, J., Wong, D.W.S., Bailey, N., Minton, J., 2018. Spatial Segregation Measures: A Methodological Review. Tijdschr. voor Econ. en Soc. Geogr. 00. doi:10.1111/tesg.12305