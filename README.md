# SRGAN

#Create Environment 

conda create -n midterm_srgan python=3.10 -y
conda activate midterm_srgan

#Install required libraries
install libraries torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn tqdm pillow

#Data structure

data/raw(original train/test dataset), sr(LR-HR paris for SRGAN training), srgan_generated(auto created later)

#For running

python train_srgan.py
python generate_sr_images.py
python classifier_train_A.py
python classifier_train_B.py
python compare_model.py
python_visualize_srgan_transforms.py





