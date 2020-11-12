---
title: 'The past and present of context variable in QGIS'
date: 2020-11-12
permalink: /posts/2020/11/The-past-and-present-of-context-variable-in-QGIS/
tags:
  - QGIS
  - python
  - context
---

The past and present of context variable in QGIS
------

如果我们要写一个QGIS中的工具箱,我们需要用Python脚本完成以下的结构,

下面的这个代码片段是QGIS给我们提供的一个例子,

```python
class Heatmap(QgisAlgorithm):
    INPUT = 'INPUT'
    RADIUS = 'RADIUS'
    RADIUS_FIELD = 'RADIUS_FIELD'
    WEIGHT_FIELD = 'WEIGHT_FIELD'
    PIXEL_SIZE = 'PIXEL_SIZE'
    KERNEL = 'KERNEL'
    DECAY = 'DECAY'
    OUTPUT_VALUE = 'OUTPUT_VALUE'
    OUTPUT = 'OUTPUT'

    def icon(self):
        return QgsApplication.getThemeIcon("/heatmap.svg")

    def tags(self):
        return self.tr('heatmap,kde,hotspot').split(',')

    def group(self):
        return self.tr('Interpolation')

    def groupId(self):
        return 'interpolation'

    def name(self):
        return 'heatmapkerneldensityestimation'

    def displayName(self):
        return self.tr('Heatmap (Kernel Density Estimation)')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
		......
    def processAlgorithm(self, parameters, context, feedback):
		......
```

我比较感兴趣的点是, 在processAlgorithm函数的参数中, context到底是什么东西？如果我们写单独运行的Python脚本的话，context该怎么构造？

这篇博客的主要目的就是探索QGIS的源代码, **追寻context构造，传递，和使用的来龙去脉** . 我们就以这个heat map工具作为例子来探索。

进过一番查找，processAlgorithm是由 gsProcessingAlgorithm::runPrepared 函数调用的，context也是从上一步传过来的。

```c++
 QVariantMap QgsProcessingAlgorithm::runPrepared( const QVariantMap &parameters, QgsProcessingContext &context, QgsProcessingFeedback *feedback )
 {
......
     QVariantMap runResults = processAlgorithm( parameters, *runContext, feedback );
......
 }
```

继续溯源。

```python
def runAlgorithm(self):
    self.feedback = self.createFeedback()
    self.context = dataobjects.createContext(self.feedback)
    ......
    task = QgsProcessingAlgRunnerTask(self.algorithm(), parameters, self.context, self.feedback)
	......
```

我们在**dataobjects.py**中，找到了createContext的函数。

```python
 def createContext(feedback=None):
     """
     Creates a default processing context
  
     :param feedback: Optional existing QgsProcessingFeedback object, or None to use a default feedback object
     :type feedback: Optional[QgsProcessingFeedback]
  
     :returns: New QgsProcessingContext object
     :rtype: QgsProcessingContext
     """
     context = QgsProcessingContext()
     context.setProject(QgsProject.instance())
     context.setFeedback(feedback)
  
     invalid_features_method = ProcessingConfig.getSetting(ProcessingConfig.FILTER_INVALID_GEOMETRIES)
     if invalid_features_method is None:
         invalid_features_method = QgsFeatureRequest.GeometryAbortOnInvalid
     context.setInvalidGeometryCheck(invalid_features_method)
  
     settings = QgsSettings()
     context.setDefaultEncoding(settings.value("/Processing/encoding", "System"))
  
     context.setExpressionContext(createExpressionContext())
  
     return context
```



```python
 def createExpressionContext():
     context = QgsExpressionContext()
     context.appendScope(QgsExpressionContextUtils.globalScope())
     context.appendScope(QgsExpressionContextUtils.projectScope(QgsProject.instance()))
  
     if iface and iface.mapCanvas():
         context.appendScope(QgsExpressionContextUtils.mapSettingsScope(iface.mapCanvas().mapSettings()))
  
     processingScope = QgsExpressionContextScope()
  
     if iface and iface.mapCanvas():
         extent = iface.mapCanvas().fullExtent()
         processingScope.setVariable('fullextent_minx', extent.xMinimum())
         processingScope.setVariable('fullextent_miny', extent.yMinimum())
         processingScope.setVariable('fullextent_maxx', extent.xMaximum())
         processingScope.setVariable('fullextent_maxy', extent.yMaximum())
  
     context.appendScope(processingScope)
     return context
```



```c++
 QgsProcessingFeedback *QgsProcessingAlgorithmDialogBase::createFeedback()
 {
   auto feedback = qgis::make_unique< QgsProcessingAlgorithmDialogFeedback >();
   connect( feedback.get(), &QgsProcessingFeedback::progressChanged, this, &QgsProcessingAlgorithmDialogBase::setPercentage );
   connect( feedback.get(), &QgsProcessingAlgorithmDialogFeedback::commandInfoPushed, this, &QgsProcessingAlgorithmDialogBase::pushCommandInfo );
   connect( feedback.get(), &QgsProcessingAlgorithmDialogFeedback::consoleInfoPushed, this, &QgsProcessingAlgorithmDialogBase::pushConsoleInfo );
   connect( feedback.get(), &QgsProcessingAlgorithmDialogFeedback::debugInfoPushed, this, &QgsProcessingAlgorithmDialogBase::pushDebugInfo );
   connect( feedback.get(), &QgsProcessingAlgorithmDialogFeedback::errorReported, this, &QgsProcessingAlgorithmDialogBase::reportError );
   connect( feedback.get(), &QgsProcessingAlgorithmDialogFeedback::infoPushed, this, &QgsProcessingAlgorithmDialogBase::pushInfo );
   connect( feedback.get(), &QgsProcessingAlgorithmDialogFeedback::progressTextChanged, this, &QgsProcessingAlgorithmDialogBase::setProgressText );
   connect( buttonCancel, &QPushButton::clicked, feedback.get(), &QgsProcessingFeedback::cancel );
   return feedback.release();
 }
```

在我们写独立脚本的时候，就不需要创造这个feedback变量了。

总结， 我们只需要添加下面的代码就可以创造context变量了

```python
from processing.tools import dataobjects
...
context = dataobjects.createContext()
...
```

