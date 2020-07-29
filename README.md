# NeuRec

## An open source neural recommender library

**Main Contributors**: [Bin Wu](https://github.com/wubinzzu), [Zhongchuan Sun](https://github.com/ZhongchuanSun), [Xiangnan He](http://staff.ustc.edu.cn/~hexn/), [Xiang Wang](https://xiangwang1223.github.io), and [Jonathan Staniforth](https://github.com/jonathanstaniforth).

**NeuRec** is a comprehensive and flexible Python library for recommender systems that includes a large range of state-of-the-art neural recommender models.
This library aims to solve general, social and sequential (i.e. next-item) recommendation tasks.
Now, NeuRec supports both [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) libraries.
NeuRec is [open source](https://opensource.org) and available under the [MIT license](https://opensource.org/licenses/MIT).

## Features

- **Flexible configuration** - easily change the configuration settings to your exact requirements;
- **Easy expansion** - quickly include models or datasets into NeuRec;
- **Fast execution** - naturally support GPU, with a multi-thread evaluator;

## Architecture

The architecture of NeuRec is shown in the below:

![Architecture](./doc/img/architecture.svg)

## Quick Start

Firstly, download this repository and unpack the downloaded source to a suitable location.

Secondly, install [RecKit](https://github.com/ZhongchuanSun/reckit):

```bash
pip install reckit
```

Thirdly, specify dataset and recommender in configuration file *NeuRec.ini*.

Finally, run [main.py](./main.py) in IDE or with command line:

```bash
python main.py
```

## Models

[check_mark]:./doc/img/check_mark.svg

The list of available models in NeuRec, along with their paper citations, are shown below:

| General Recommender | PyTorch | TensorFlow | Paper                                                                   |
|---|:-:|:-:|---|
| BPRMF     |   ![√][check_mark]   | ![√][check_mark]  | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.    |

| Sequential Recommender | PyTorch | TensorFlow | Paper                                                                   |
|---|:-:|:-:|---|
|      |     |      |      |

| Social Recommender | PyTorch | TensorFlow | Paper                                                                   |
|---|:-:|:-:|---|
|      |     |      |      |

## Contributions

Please let us know if you experience any issues or have suggestions for new features by submitting an issue under the Issues tab or joining our WeChat Group.

![WeChat](./doc/img/WeChat_QR_Code_256.jpg)

## Acknowledgements

The development of NeuRec is supported by the National Natural Science
Foundation of China under Grant No. 61772475. This project is also supported by the National Research Foundation, Prime Minister’s Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="./doc/img/next.png" width = "297" height = "100" alt="NEXT++" align=center />
