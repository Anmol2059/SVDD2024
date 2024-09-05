## Implementation of [SPEECH FOUNDATION MODEL ENSEMBLES FOR THE CONTROLLED SINGING VOICE DEEPFAKE DETECTION CTRSVDD CHALLENGE 2024](https://arxiv.org/pdf/2409.02302), Part of SVDD Challenge

### Steps to Implement:

1. **Prepare the Dataset:**
   - Follow the instructions in the [SVDD Challenge CtrSVDD_Utils Repository](https://github.com/SVDDChallenge/CtrSVDD_Utils) to manage the dataset.

2. **Organize Dataset Structure:**
   - Ensure your main directory has the following structure:
     ```
     Datasets/
     ├── dev/
     ├── train/
     ├── eval/
     ├── dev.txt
     └── eval.txt
     ```

3. **Set Up the Environment:**
   - Create a conda environment using the provided `requirements.txt` file:
     ```sh
     conda create --name your_env_name --file requirements.txt
     conda activate your_env_name
     ```

4. **Run Training:**
   - Execute the training script by specifying the base directory of the dataset:
     ```sh
     python train.py --base_dir {path_to_Datasets_folder}
     ```
   - Additional arguments can be added, such as `--algo` for the rawboost algorithm:
     ```sh
     python train.py --base_dir {path_to_Datasets_folder} --algo {algorithm_choice}
     ```
   - To change the model, modify the model selection directly in the `train.py` script header.

5. **Run Evaluation:**
   - Execute the evaluation script by specifying the base directory of the dataset:
     ```sh
     python eval.py --base_dir {path_to_Datasets_folder}
     ```

### Additional Information:

- **Custom Arguments:**
  - You can customize various parameters through command-line arguments as needed.
  - Example:
    ```sh
    python train.py --base_dir {path_to_Datasets_folder} --batch_size 64 --epochs 50
    ```

- **Changing the Model:**
  - To use a different model, edit the model import and instantiation in the `train.py` file.

For further details, refer to the  code comments within the scripts.
