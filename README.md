# Introduction
With the trend of applying deep learning (DL) in high performance scientific computing, the unique characteristics of emerging DL workloads in HPC raise great challenges in designing, implementing HPC AI systems. The community needs a new yard stick for evaluating the future HPC systems. we propose HPC AI500 --- a benchmark suite for evaluating HPC systems running scientific DL workloads. Covering the most representative scientific fields, each workload from HPC AI500 is based on real-world scientific DL applications.

# Scientific Deep Learning
Recent years, DL has replaced traditional scientific computing methods and becomes a promising tool in several scientific computing fields especially in extreme weather analysis, cosmology, and high energy physics.  

## Extreme Weather Analysis (EWA)
![EWA](./images/ewa.png)

## Cosmology (Cos)
![COSMOLOGY](./images/cosmology.png)

## High Engergy Physics (HEP)
![HEP](./images/hep.png)

# Benchmarking Methodology
As HPC AI is an emerging and evolving domain, we take an incremental and iterative approach.
![MYTH](./images/methodology.png)

For workloads diversity, we map the scientific deep learning task to classical AI problems. 

| Fileds | DL Apps | AI Problems |
| --- | --- | --- |
| EWA | Identify extreme weather patterns | Object detection |
| HEP | Jet-images discrimination | Image recognition |
| Cosmology | Estimate parameters, Galaxy images generation | Image recognition, Image generation |


For dataset diversity, cause the main data schema is matrix, we classify the matrix to 3 categories and choose 3 representative dataset according to real scientific deep learning applications. 

| Fields | Data Format | Features |
| --- | --- | --- |
| EWA | 2D dense matrix | High resolution, 16 channels |
| HEP | 2D sparse matrix | 3 channels (not RGB), sparse |
| Cosmology | 3D matrix | 3 dimensional  |

# HPC AI500 Benchmark
Currently, the overview of HPC AI500 benchmark is shown in the table below.

| Scenarios | Workloads | Involved field | Datasets | Software <br>stack |
| --- | --- | --- | --- | --- |
| Micro benchmarks | <font size="3">Convolution Pooling <br>Fully-connected</font> | N/A | Matrix | MKL <br>CUDNN |
| Image recognition | ResNet | HEP Cosmology | Paticle collision datset, N-body dataset | TensorFlow<br>PyTorch |
| Object detection | Faster-RCNN | EWA | CAM5 dataset |  TensorFlow<br>PyTorch|
| Image generation | DCGAN | Cosmology | N-body dataset |  TensorFlow<br>PyTorch|


## Metric
### Component Benchmark
At present, <b>Time-to-Accuracy</b> is the most well-received solution(e.g. DAWNBench and MLPerf). For comprehensive evaluate, the training accuracy and validation accuracy are both provided. The former is used to measure the training effect of the model, and the latter is used to measure the generalization ability of the model. The threshold of target accuracy is defined as a value according to the requirement of corresponding application domains. Each application domain needs to define its own target accuracy. In addition, cost-to-accuracy and power-to-accuracy are provided to measure the money and power spending of training the model to the target accuracy.

### Micro Benchmark
The metrics of the micro benchmarks is simple since we only measure the performance without considering accuracy. we adopt <b> FLOPS </b>and images per second(images/s) as two main metrics. We also consider power and cost related metrics.

# Comparison of AI Benmarking Efforts
Most of the existing AI benchmarks are based on commercial scenarios. Deep500 is a benchmarking framework aiming to evaluate high-performance deep learning. However, its reference implementation uses commercial open source data sets and simple DL models, hence cannot reflect real-world HPC AI workloads. We summary these major benchmarking efforts for AI and compare them with HPC AI500 as shown in the table below.

<table>
     <TR>
       <TD ROWSPAN="3"><b>Benchmarks</b></TD>
       <TD ROWSPAN="3"><b>Datasets</b></TD>
       <TD COLSPAN="4"><b>Problems domains</b></TD>
       <TD COLSPAN="2"><b>Implementation</b></TD>
     </TR>
     <TR>
       <TD COLSPAN="3"><b>Scientific</b></TD>
       <TD ROWSPAN="2"><b>Commercial</b></TD>
       <TD ROWSPAN="2"><b>Standalone</b></TD> 
       <TD ROWSPAN="2"><b>Distributed</b></TD>  
     </TR>
     <TR>
       <TD><b>EWA</b></TD>
       <TD><b>Cos</b></TD>
       <TD><b>HEP</b></TD>
     </TR>
     <TR>
       <TD>HPC AI500</TD>
       <TD>Scientific data</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
       <TD>&#10005</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
     </TR>
     <TR>
       <TD>TBD</TD>
       <TD ROWSPAN="5">Commercial data</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
     </TR>
      <TR>
       <TD>MLPerf</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
       <TD>&#10005</TD>
     </TR>
      <TR>
       <TD>DAWNBench</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
       <TD>&#10005</TD>
     </TR>
      <TR>
       <TD>Fathom</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10005</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
       <TD>&#10005</TD>
     </TR>
      <TR>
       <TD>Deep500</TD>
       <TD COLSPAN=4>Framework, undefined</TD>
       <TD>&#10003</TD>
       <TD>&#10003</TD>
     </TR>
</table>

# Refference Implementation
Currently, we provide a scalable reference implementation of EWA workloads, see <a href="http://125.39.136.212:8090/XW.Xiong/EWA">EWA workload</a>.
# Contributing
HPC AI500 is an open-source project. We are looking forward to accept Pull Request to improve this project.

# License