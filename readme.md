* Download `git clone git@github.com:xiang-yu/Carleman-QLBM.git` into your home directory
* `vi ~/.bashrc` and add the following,
```bash
export QCFD_HOME=$HOME/Carleman-QLBM/
export QCFD_SRC=$QCFD_HOME/src/
```
then
`cd QCFD_SRC/CLBM`

`Julia`

```julia
julia> using Pkg
julia> Pkg.activate("../..")  # Activate the main project environment from src/CLBM
julia> Pkg.instantiate()
julia> include("clbm_run.jl")       
```
