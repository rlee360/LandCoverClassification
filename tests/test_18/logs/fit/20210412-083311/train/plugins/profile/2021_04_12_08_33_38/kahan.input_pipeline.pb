	&??:*ԍ@&??:*ԍ@!&??:*ԍ@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-&??:*ԍ@???!?V@1?%ZR??@A??/??I??]?C??*	?rh?}\?@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2}?Жs9A@!e??8X@)}?Жs9A@1e??8X@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2;?I/???!?k?=6C??);?I/???1?k?=6C??:Preprocessing2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::TensorSlice??:????!??j??B??)??:????1??j??B??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch	m9?⪲?!?=??<@??)	m9?⪲?1?=??<@??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??bFx{??!??de?-??)??bFx{??1??de?-??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismס???ù?!??	???)9ӄ?'c??1???????:Preprocessing2F
Iterator::Model&??:????!??????)uZ?A????17?6?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?7l?{???Q?'?+?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???!?V@???!?V@!???!?V@      ??!       "	?%ZR??@?%ZR??@!?%ZR??@*      ??!       2	??/????/??!??/??:	??]?C????]?C??!??]?C??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?7l?{???y?'?+?X@