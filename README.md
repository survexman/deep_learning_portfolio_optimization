`ga` (read as "ga") is a Python package connecting genetic algorithms optimization and deep learning. Its goal is to
facilitate research of networks that perform weight allocation in **one forward pass**.

# Installation

```bash
pip install ga
```

# Resources

- [**Getting
  started**](https://www.amazon.com/Learning-Genetic-Algorithms-Python-capabilities-ebook/dp/B08WKBFHGV/?_encoding=UTF8&pd_rd_w=XehOg&content-id=amzn1.sym.cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_p=cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_r=137-5598848-0173063&pd_rd_wg=ojbqc&pd_rd_r=58b3535d-97fc-4759-b1f7-8389d6420821&ref_=aufs_ap_sc_dsk)
- [**Detailed
  documentation**](https://www.amazon.com/Learning-Genetic-Algorithms-Python-capabilities-ebook/dp/B08WKBFHGV/?_encoding=UTF8&pd_rd_w=XehOg&content-id=amzn1.sym.cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_p=cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_r=137-5598848-0173063&pd_rd_wg=ojbqc&pd_rd_r=58b3535d-97fc-4759-b1f7-8389d6420821&ref_=aufs_ap_sc_dsk)
- [**More
  examples**](https://www.amazon.com/Learning-Genetic-Algorithms-Python-capabilities-ebook/dp/B08WKBFHGV/?_encoding=UTF8&pd_rd_w=XehOg&content-id=amzn1.sym.cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_p=cf86ec3a-68a6-43e9-8115-04171136930a&pf_rd_r=137-5598848-0173063&pd_rd_wg=ojbqc&pd_rd_r=58b3535d-97fc-4759-b1f7-8389d6420821&ref_=aufs_ap_sc_dsk)

# Description

`deepdow` attempts to **merge** two very common steps in portfolio optimization

1. Forecasting of future evolution of the market (LSTM, GARCH,...)
2. Optimization problem design and solution (convex optimization, ...)

It does so by constructing a pipeline of layers. The last layer performs the allocation and all the previous ones serve
as feature extractors. The overall network is **fully differentiable** and one can optimize its parameters by gradient
descent algorithms.

# `deepdow` is not ...

- focused on active trading strategies, it only finds allocations to be held over some horizon (**buy and hold**)
    - one implication is that transaction costs associated with frequent, short-term trades, will not be a primary
      concern
- a reinforcement learning framework, however, one might easily reuse `deepdow` layers in other deep learning
  applications
- a single algorithm, instead, it is a framework that allows for easy experimentation with powerful building blocks

# Some features

- all layers built on `torch` and fully differentiable
- integrates differentiable convex optimization (`cvxpylayers`)
- implements clustering based portfolio allocation algorithms
- multiple dataloading strategies (`RigidDataLoader`, `FlexibleDataLoader`)
- integration with `mlflow` and `tensorboard` via callbacks
- provides variety of losses like sharpe ratio, maximum drawdown, ...
- simple to extend and customize
- CPU and GPU support

# Citing

If you use `deepdow` (including ideas proposed in the documentation, examples and tests) in your research please **make
sure to cite it**.
To obtain all the necessary citing information, click on the **DOI badge** at the beginning of this README and you will
be automatically redirected to an external website.
Note that we are currently using [Zenodo](https://zenodo.org/).