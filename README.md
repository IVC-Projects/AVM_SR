# AVM\_SR&#x20;

# Dependency &#x20;

*   Python 3.6
*   PyTorch
*   glob
*   numpy
*   pillow
*   tqdm

# TestSet

*   Use the div2k dataset in YUV format for training, and the test sequence is stored in the `./Testset` directory

<!---->

*   `./DIV2K_val_LR_AVM` directory stores low resolution images obtained from the AVM encoder that have not been super-resolutioned.

# transfer

**1. The converted LUT file will be saved in `\transfer\SPLUT_0227`**

The stored files are all converted table files that can be run outside of the torch environment.



**2. Arb SR Checkpoints will be saved in `.\transfer\ArbSR_0423`**

The **Checkpoints** stored here are dependent on the torch environment

# Super-Resolution Using Look-Up Table

> python Inference\_SPLUT\_M\_YUV\_rot.py

Based on SPLUT, we optimized the network structure and attempted to correspond to tables with each QP.



For example, `LUT1_K122_135_Model_S_A.npy`, **LUT1** represents a stage, **K122** represents the shape of the table, **135** represents the Qp range, and the final **A** represents branch A.


# Arbitrary Super-Resolution

> python Inference_arbSR.py



The network structure of the model exists in the `./models` folder