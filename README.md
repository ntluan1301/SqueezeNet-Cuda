# SqueezeNet-Cuda
Implement SqueezeNet from scratch in CUDA (including forward/backward passes and optimizer), achieving 90% accuracy, a 1000× speedup over NumPy, and 0.017 s inference time per 3×224×224 image.

## Explanation of the Functions of the Files

### 1. **File: Cuda_Adam_Tunning_Parameter.ipynb**
   - **Description**: Contains CUDA code to build the SqueezeNet model and use the Adam optimizer to fine-tune parameters after 5 epochs.
   - **Function**:
     - Helps the model overcome the initial difficult phase that SGD cannot learn from.
     - Optimizes initial parameters thanks to Adam's adaptive learning rate characteristics.
     - Creates a high-quality checkpoint (`model_fintuning_by_Adam.npz`) for use in subsequent training phases.

### 2. **File: Cuda_Training_SGD.ipynb**
   - **Description**: Contains CUDA code to build the SqueezeNet model and use the SGD optimizer for training, initializing from parameters that have been fine-tuned by Adam.
   - **Function**:
     - Continues training with parameters optimized by Adam, leveraging SGD to achieve higher accuracy and better convergence.
     - Combines the advantages of Adam (initial fine-tuning) and SGD (long-term training).
### Note: Reasons for Using SGD and Adam Combination
   - **Reason 1**: SGD encounters difficulties in the early phase, unable to overcome non-learning regions.
   - **Reason 2**: Adam after 5 epochs helps the model overcome the difficult phase, but from epoch 5 onward, the group's Adam implementation encounters gradient vanishing issues, reducing accuracy.
   - **Reason 3**: Our SGD implementation ensures 100% compatibility with PyTorch, allowing strong model learning after overcoming the difficult phase thanks to Adam.
## Usage Instructions
**Note**: A GPU is required to run these files.

### **Retraining from Scratch**
   - **Step 1**: Download the dataset from: [Tomato Diseases Dataset](https://www.kaggle.com/datasets/luisolazo/tomato-diseases).
   - **Step 2**: In the file `Cuda_Adam_Tunning_Parameter.ipynb`, replace the path `/kaggle/input/tomato-diseases` with the path to the downloaded dataset.
   - **Step 3**: Run the file `Cuda_Adam_Tunning_Parameter.ipynb` to fine-tune parameters using Adam.
   - **Step 4**: Find the checkpoint `model_fintuning_by_Adam.npz` saved by the function `np.savez(f'model_fintuning_by_Adam.npz', weights=weights_np, m_adam=M_adam_np, v_adam=V_adam_np)` in the above file.
   - **Step 5**: In the file `Cuda_Training_SGD.ipynb`:
     - Replace the path `/kaggle/input/tomato-diseases` with the path to the dataset.
     - Replace the path `/kaggle/input/epoch_30_sgd/other/default/1/model_data_SGD_epoch_25.npz` with `model_fintuning_by_Adam.npz` from step 4.
     - Run the file to continue training with SGD.
## Notes
- Ensure the environment has a GPU to run the files.
- Carefully check file paths and dataset before running.
- Detailed results and performance charts are saved in the file `Presentation - SqueezeNet for Leaf Disease Detection.pdf`
