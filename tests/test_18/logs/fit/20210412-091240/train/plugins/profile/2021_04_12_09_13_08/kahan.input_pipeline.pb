	??R?w?@??R?w?@!??R?w?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??R?w?@?z?Fw`@1?ng_???@A??^~?ɸ?I?6???@*	?K7?I??@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2????)@@!A???Z?W@)????)@@1A???Z?W@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2/???uR??!p??
??@)/???uR??1p??
??@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??ޫV&??!2s5?]???)??ޫV&??12s5?]???:Preprocessing2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::ParallelMapV2::TensorSlicew??/???!??[?d??)w??/???1??[?d??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?!??gx??!???*???)?!??gx??1???*???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?ͮ{+??!h~?U??)x???N̚?1(
\??˳?:Preprocessing2F
Iterator::Model????e??!?&4?=???)??c> Љ?1?A????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ???a3??QL?x2?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?z?Fw`@?z?Fw`@!?z?Fw`@      ??!       "	?ng_???@?ng_???@!?ng_???@*      ??!       2	??^~?ɸ???^~?ɸ?!??^~?ɸ?:	?6???@?6???@!?6???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ???a3??yL?x2?X@