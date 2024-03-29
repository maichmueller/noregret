# <p align=center> no-regret

 [![Linux C++](https://github.com/maichmueller/noregret/actions/workflows/cpp.yml/badge.svg)](https://github.com/maichmueller/noregret/actions/workflows/cpp.yml)

---

## <p align=center> noregret is a performance-built c++ framework library implementing regret minimization algorithms for [Factored-Observation Stochastic Games (FOSG)](https://www.sciencedirect.com/science/article/pii/S000437022100196X).
Game environments may utilize their own implementation designs as long as they provide an interface agreeing with the FOSG formalism defined by constraints on types.
This framework makes heavy use of c++20's concepts to constrain environments to specifc algorithm's needs without forcing the same details everywhere.

### Currently implemented:

*algorithms*:

- [**Vanilla CFR**](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf) (alt / sim)

- [**CFR+**](https://www.science.org/doi/full/10.1126/science.1259433?casa_token=3o6gJN7ICksAAAAA:TmUKYNEs7BqQEV2yrRdNJ5OJrdPNA-MAzwJsS88B3M5lRB2iORiiBQBepozAi85M5tY-FLE_rGir8nQ)

- [**Discounted CFR**](https://ojs.aaai.org/index.php/AAAI/article/view/4007) (alt / sim)

- [**Linear CFR**](https://ojs.aaai.org/index.php/AAAI/article/view/4007) (alt / sim)

- [**Exponential CFR**](https://arxiv.org/abs/2008.02679) (alt / sim)

- [**Monte-Carlo CFR**](https://papers.nips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)
    - [**Outcome Sampling**](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf) (lazy weighting / stochastic weighting - alt / sim)
    - [**External Sampling**](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf) (stochastic weighting)
    - [**Chance Sampling**](https://www.ma.imperial.ac.uk/~dturaev/neller-lanctot.pdf) (alt / sim)

- [**Pure CFR**](https://richardggibson.appspot.com/static/work/thesis-phd/thesis-phd-paper.pdf) (alt / sim)

*environments*:

- **Rock Paper Scissors**
- **Kuhn Poker**
- **Leduc Poker**
- **Stratego**

## Installation

To install the _noregret_ library, you can follow the steps below. Note that the library is an interface (pure header)
library and so does not need building. You may choose to build it anyway to test it against your system.

### Prerequisites

Make sure you have the following dependencies installed on your system:

    CMake (version 3.17 or higher)
    Conan (version 1.xx)

### Steps

Step 1:
Clone the no-regret repository to your local machine:
```bash
git clone https://github.com/maichmueller/noregret.git
cd noregret
```
Step 2 (Optional): Configure, build, and run tests

Configure
```bash
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
```
CMake will call Conan for you to install the dependencies.
Ensure that the Conan executable can be found in your **PATH**!

Build
```bash
cmake --build ./build --config Release --target nor_tests_all
```
Run
```bash
ctest Test_nor_tests_all
```

Step 4: Install the Library (Optional)

If you want to install the library system-wide, you can use the following command:

bash

cmake --install .

By default, the library will be installed in the appropriate system directories based on your platform.
If you want to specify a custom installation directory, you can use the --prefix option followed by the desired path.

Once the installation is complete, you can use noregret as library in your C++ projects by including
the necessary headers and linking against the installed library.

For more information on how to use the library, currently please refer to the tests provided in the repository.
