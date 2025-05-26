## Sci-Tech Code Test Report

### Introduction

The project successfully developed an LSTM model to predict reentry timestamps on an unseen validation dataset extracted from the training data. A straightforward LSTM model, combined with feature engineering, demonstrated strong performance. Due to the relatively small training dataset, 20% of the data was allocated for validation without being used in model training. In the fully trained model, validation errors did not exceed 0.50 seconds (one time step). Inference results are also stored in a CSV file within the logs folder along with the inference log.  

Given the training dataset's size of 90 segments, the validation set comprised 18 missile launch and reentry events. None of the 18 validation segments exhibited errors greater than 0.50 seconds (one timestamp), leading to a validation RMSE of 0.118 seconds. In the most recent training run, only one track showed an error, whereas previous runs typically exhibited errors in 1-6 tracks, though never exceeding 0.50 seconds. In comparison, a baseline method using average altitude for reentry resulted in an RMSE of 3.659 seconds, with errors reaching up to 7 seconds. The LSTM model significantly outperformed this baseline approach.  

![Validation Error](img/val_error_hist_final.png)  
![Validation Error](img/baseline_error_hhistogram.png)  

The dataset exhibits significant noise; however, the model effectively navigated this challenge through feature engineering, incorporating additional data derived from combinations of the provided inputs. Given more time, I would experiment with moving averages to further smooth the noise. However, considering the model's high accuracy, there is limited room for improvement on the available validation data.  

### Feature Engineering

Data analysis and feature engineering were conducted using a Jupyter notebook included in the project (SciTechDataStudy.ipynb). The initial set of features provided for the model was minimal, but exploratory analysis using the Jupyter notebook suggested several potential feature enhancements.
The first extracted feature was altitude change, analyzed on a track-wise basis. A histogram of altitude change revealed that introducing a descent flag could be beneficial when paired with a descent threshold. Despite some noise in altitude change, distinct gaps in the histogram allowed the descent threshold to function effectively for an "is_descending" flag. By plotting the data, it was confirmed that all reentry events in the training dataset occurred during the descending phase—an insight made practical by the dataset's size.  

To compute horizontal distance, latitude and longitude values were processed using the haversine formula for great-circle distance. Additionally, total distance—including altitude change—was calculated. Combined with the fixed time step delta of 0.5 seconds, this enabled the derivation of a velocity feature. These engineered features improved the model’s performance during initial testing and were retained in the final version.  

Certain features were removed to streamline the model, including the sensor ID column, which contained only a single unique value in the training data. Timestamps were uniformly spaced, ensuring compatibility with the LSTM model, so no imputation of missing data was necessary. Although timestamp was not used as an input feature, it was retained for output data reporting.  

The entire sequence for each track ID was passed to the model. As a potential enhancement, the model could have been adjusted to focus solely on the descent phase of the data. While this approach was tested and performed well, it was ultimately discarded due to dependencies on flight profiles—particularly in cases of potential failed launches, where descent phase characteristics could differ.  

### The LSTM Model  

A relatively simple LSTM model was implemented, consisting of two LSTM layers and a linear output layer. Initially intended as a baseline model, it performed exceptionally well and was retained with adjustments to further optimize its performance. Each LSTM layer contained two internal layers. Attempts to improve performance by increasing model complexity with dropout and batch normalization did not yield better results on the validation dataset. Through experimentation, 260 epochs were chosen for training, and early stopping did not enhance results—the model consistently performed best toward the end of training. In some cases, early stopping prematurely halted training, leading to decreased validation performance. Cosine annealing was applied to gradually reduce the learning rate throughout training, and a lower learning rate near the final epochs appeared to improve overall model performance.  

Although the assignment called for batch normalization and dropout, both were tested and ultimately excluded from the final model due to their negative impact on performance. Early stopping was also recommended but proved ineffective, as the model consistently achieved its best results when trained to completion. Given that this is a shallow network with only two LSTM layers and a linear output layer, batch normalization is generally more beneficial in deeper architectures.  

If a larger dataset and multiple GPUs were available, training could be distributed using PyTorch DDP (Distributed Data Parallel). In such cases, PyTorch’s synchronized batch normalization layer (torch.nn.SyncBatchNorm) would be necessary to ensure consistent normalization across processes during training.

### Monitoring

The training process is monitored using a standard Python log file, which includes a header detailing the training setup. During each epoch, key metrics such as epoch number, training loss, validation loss, learning rate, and epoch duration are recorded. Once training is complete, the best epoch is logged alongside the total training time.  

After training, a set of inferences is executed, with inference times recorded in the log. Results for each validation track are logged in a structured format, enabling easy extraction into a CSV file for further analysis. All log entries are timestamped and include log levels. Additionally, RMSE is calculated for true vs. predicted reentry times and logged. The total duration for the entire process is logged at the end of the file. An example training log file is available within the Python project files.  

The inference process follows a similar logging structure. The log file includes a header, ensuring consistency with the training logs. Inference results are formatted to facilitate CSV extraction, and a CSV file is generated for analysis. Each inference track is logged with key fields: Track ID, Reentry Index, Reentry Time, and Inference Run Time. The total inference run time is also recorded as the final log entry.  

### Commentary on Coding 

Robustness in the inference script was ensured by wrapping critical functionality in try-except blocks. No exceptions have been observed with this dataset, likely due to its cleanliness and absence of missing entries. The inference script employs multi-processing for scalability; however, given the small size of the test dataset, this approach is actually slower than processing sequentially in a loop. Therefore, scaling to a sufficiently large number of predictions would be necessary to justify the overhead of multi-processing.  

Logging is implemented using the standard Python logger set to the "info" level. A few debug statements were used during development and can be enabled when needed. Exceptions are captured and logged with a stack trace for debugging and analysis. Since inference is multi-processed, exceptions in sub-processes are handled gracefully—they are logged without disrupting other processes. Failed tracks are recorded with error messages, while successful track IDs continue processing and generate output as expected. The importance of robust error logging was confirmed through inevitable mistakes encountered during development.  

Backup management was handled by periodically copying the working code folder to a flash drive. In a professional setting, version control would ideally be managed via Git, with changes committed to a dedicated branch. The code was formatted using Black and validated with MyPy. Unit testing was conducted using pytest. While a production-grade implementation would benefit from a configuration file, time constraints led to handling configurations through constants defined in Python files.  

The only file subjected to unit testing was utils.py, achieving a test coverage of 87%. The testing process included verifying that the LSTM network's forward function executes without errors, which was accomplished using the torchinfo package. Most of the functionality in train.py and inference.py depends on running training on a GPU, making unit testing in a production environment challenging. Based on my experience, these types of functions are difficult to test effectively without dedicated hardware. At my previous workplace, Bitbucket automatically ran unit tests as part of merging changes into the main branch of the Git repository. However, since a GPU was not available to BitBucket, these types of functions were not tested.  

### Project Structure

The Python training script (app_train.py) can be executed from the top level of the project. It calls train.py and utilizes shared utility functions from utils.py. Similarly, inference is performed using app_inference.py, which follows the same execution structure.  

Both application scripts (app_train.py and app_inference.py) serve as entry points, containing only code to invoke the main function of the relevant application, located in the src folder. The utils.py file, placed in the lib folder, centralizes shared functionality to streamline development and maintain modularity.  

![Validation Error](img/project_structure.png)

### Running the Project without a Container  

The project can be run from the command line with a compatible Python conda environment and needed packages installed. Run it from the root folder of the project as follows:  

+python app_train.py (to run a training session, GPU needed)  
+python app_inference.py (to run an inference session, training is a prerequisite, no GPU needed)

The project has been tested in Linux with Ubuntu 24.04. While it is forgiving of a range of python packages, below is a list of the packages that the application was tested on:  

Python                    3.10.16  
torch                     2.7.0+cu118  
torchaudio                2.7.0+cu118  
torchinfo                 1.8.0  
torchvision               0.22.0+cu118  
pandas                    2.2.3  
numpy                     2.2.5  
scipy                     1.15.3  
scikit-learn              1.6.1  
matplotlib                3.10.0  
black                     25.1.0  
pytest                    8.3.5  
pytest-cov                6.1.1  
mypy                      1.15.0  
mypy-extensions           1.1.0  

### Containerization 

Deployment was successfully completed using a Docker container, and the code was fully tested within this environment. Both training and inference performed equally well in the container as they did in Linux command-line testing. The Dockerfile in the project provides a complete recipe for building the Docker image.
However, two challenges were encountered:  

1. Large container size with GPU support – The Docker container supporting GPU acceleration was too large to send via email. Based on information from OpenAI's ChatGPT (OpenAI, 2024), reducing the container size significantly appears difficult, as even a minimal GPU runtime container could be around 5GB. The project utilizes an official PyTorch Docker distribution, which may contribute to the size. With more Docker expertise, a smaller container might be achievable, but ChatGPT was skeptical about reducing it below 25MB for email transfer.  
    
2. Minikube installation issues – Minikube proved difficult to install or was incompatible with my Ubuntu 24.04 Linux distribution, preventing successful installation. I have no prior experience with Minikube.  

Despite these issues, I am confident that with some mentorship, these challenges could be easily resolved. Sci-Tech already has working container implementations, and I am certain I could quickly develop expertise in this area—especially considering my successful experience building a Docker-compatible PyTorch GPU environment.
The deployment recipe is documented in app_train.py and follows this approach:  

+docker pull pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime  
+docker build -t pytorch-app .  
+docker run -it --rm --gpus all -v \\$(pwd)/logs:/app/logs -v \\$(pwd)/model:/app/model -p 8898:8898 pytorch-app (gets a bash terminal)  
+python app_train.py (in the bash terminal)  
+python app_inference.py (in the bash terminal)  

### Summary

The model and project were largely successful, with the only challenges being the absence of Minikube and the difficulty of constructing a GPU-supported Docker image small enough for email transfer. These issues could likely be resolved with mentorship and access to a working system example.  

During development, ChatGPT (OpenAI, 2024) and Microsoft Copilot (Microsoft, 2025) were consulted. Exploring a larger LSTM architecture could be beneficial, though it may require batch normalization and/or dropout for stability. Given the strong performance of the current model, demonstrating a statistically significant improvement in validation results would require multiple training runs and thorough testing across different train-validation splits of the dataset, or more data.  

Feature engineering played a crucial role in enhancing the model, ultimately leading to an effective LSTM-based approach for predicting the reentry phase timing of flight.    

### Citations

Microsoft. (2025). Copilot (GPT-4) [Large Language Model]. [Copilot website.](https://copilot.microsoft.com/chats/M94K2i5TFktPWDrA67n5Q)  

OpenAI. (2024). ChatGPT (May 24 version) [Large language model]. https://chat.openai.com/

