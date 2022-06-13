# Differentiable Transient Rendering

[Project Page](http://vclab.kaist.ac.kr/siggraphasia2021/index.html) | [Paper](http://vclab.kaist.ac.kr/siggraphasia2021/dtrensient-rendering-main.pdf) | [Presentation](https://www.youtube.com/watch?v=s62IiuZjBBQ)

Authors: Shinyoung Yi (syyi@vclab.kaist.ac.kr), Donggun Kim (dgkim@vclab.kaist.ac.kr),  Kiseok Choi (kschoi@vclab.kaist.ac.kr), Adrian Jarabo, Diego Gutierrez, Min H. Kim (minhkim@kaist.ac.kr)

Institute: KAIST Visual Computing Laboratory

If you use our code for your academic work, please cite our paper:

```
@Article{ShinyoungYi:SIGA:2021,
  author  = {Yi, Shinyoung and Kim, Donggun and Choi, Kiseok and 
             Jarabo, Adrian and Gutierrez, Diego and Kim, Min H.},
  title   = {Differentiable Transient Rendering},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2021)},
  year    = {2021},
  volume  = {40},
  number  = {6},
  }    
```

## Installation
Our *differentiable transient renderer* was implemented based on *Path-space Differentiable Renderer (PSDR)* [[Zhang et al. 2020]](https://shuangz.com/projects/psdr-sg20/). You can install our source codes in the flowing steps. We provide two ways to run our code, by general installation (on local environment) or by Docker (virtual environment, fully automatic install).

### General Installation

1. First obtain the entire source codes of the CPU-version of PSDR from their webpage](https://shuangz.com/projects/psdr-sg20/) and unzip the ZIP file.
2. Rename `pypsdr` folder to `pydtrr`.
3. Copy all the files in our repository and paste to the unziped directory. Then remained installation steps are similar to PSDR.

4. Install all the dependencies with the provided script. If you are using a Docker virtual environment, use `./install_docker.sh` instead of `./install.sh`.

   ```
   ./install.sh
   exec bash -l
   ```

5. Compile and install the `dtrr` & `pydtrr` library. Here, `{#param}` denote the number of scene parameters you want to change (default: 1).

   ```
   mkdir build
   cd build
   cmake .. -D_NDER={#param}
   sudo make install -j
   ```

### Using Docker

If you have concerns about your local environment and want to run code in a separate environment, we recommend using this procedure. Also, this procedure automatically installs all dependencies at once.

1. Build docker image using provided Dockerfile.

   ```
   docker build -t dtrr:latest .
   ```

2. Generate docker container with image built at previous step. ( `docker run` or `docker start` ) Inside docker container, go to directory where this repository exists.

3. Run install script. Note that you should run `install_docker.sh`. This will automatically install all dependencies and perform step1-4 in general installation.

   ```
   ./install_docker.sh
   ```

   You can see where it is installed when install is finished. Make sure you are at that installed directory, and follow next step.

4. Compile and install the `dtrr` & `pydtrr` library. Here, `{#param}` denote the number of scene parameters you want to change (default: 1).

   ```
   mkdir build
   cd build
   cmake .. -D_NDER={#param}
   make install -j
   ```

## Dependencies

Our source codes inherit the dependency of PSDR.
- [Eigen3](http://eigen.tuxfamily.org)
- [Python 3.6 or above](https://www.python.org)
- [pybind11](https://github.com/pybind/pybind11)
- [PyTorch 1.0 or above](https://pytorch.org)
- [OpenEXR](https://github.com/openexr/openexr)
- [Embree](https://embree.github.io)
- [OpenEXR Python](https://github.com/jamesbowman/openexrpython)
- A few other python packages: numpy, scikit-image, matplotlib


## Usage
We provide the script to reproduce the results of paper example [Egg].

To generate transient images and transient derivatives of *Egg* scene (Fig. 4 left in our paper), at `scenes/egg`,

```
python3 run.py -spp={spp}
```
Here, `{spp}` denotes samples per pixel so that our code computes `{spp}` * (image width) * (image height) * (number of transient frames) light paths. The default value for this script is `{spp}=16`, which takes about 29 min.

Our code was tested on Intel i9-10920X CPU with Ubuntu 20.04.

## License

Shinyoung Yi, Donggun Kim, and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.

The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].

Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.

NB Our implementation is covered under the license of "Path-space differentiable rendering" codes (BSD 3-Clause "New" or "Revised" License).

Please refer to license.txt for more details. 

## Contact

If you have any questions, please feel free to contact us.

Shinyoung Yi (syyi@vclab.kaist.ac.kr)

Donggun Kim (dgkim@vclab.kaist.ac.kr)

Min H. Kim (minhkim@vclab.kaist.ac.kr)