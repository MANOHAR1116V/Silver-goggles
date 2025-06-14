# Silver-goggles

**1. **Imports & Setup****
Pulls in the standard libraries you’ll need for data wrangling (os, numpy, pandas), plotting (matplotlib, seaborn), image handling (PIL), deep‑learning (tensorflow.keras), and classic ML utilities (sklearn). This is the toolbox for everything that follows.

**2. Collecting File Paths & Labels**
root_dir — folder that contains one sub‑folder per class (e.g. NonDemented/, MildDemented/, …).

Loops through each sub‑folder, grabs every *.jpg / *.jpeg / *.png, and records two things in a list of dicts:

filename — relative path like NonDemented/img1.jpg

label — the sub‑folder name (i.e., the class).

Converts that list into a pandas DataFrame (df) and writes it to image_labels.csv for later inspection.

**3. Train / Validation / Test Split**
Uses train_test_split twice so the final proportions are:

60 % train

20 % validation

20 % test
Stratification keeps class ratios the same across all three sets.

**4. ImageDataGenerators**
Training generator (train_datagen) — applies heavy augmentation (rotation, shift, shear, zoom, flip) and rescales pixels to [0, 1].

Validation/Test generator (val_test_datagen) — only rescales.

Flow‑from‑dataframe calls turn the splits into iterable generators that automatically read, resize to 224×224, batch, shuffle (except test), and one‑hot‑encode labels.

**5. Class‑Weight Calculation**
Computes balanced class weights so rarer classes receive larger loss penalties during training, mitigating class imbalance.

**6. Model Architecture & Fine‑Tuning**
Backbone: MobileNetV2 pretrained on ImageNet, with the top removed.

Layer Freezing: freezes everything except the last 50 layers (transfer‑learning sweet‑spot).

Head:

GlobalAveragePooling2D → flattens spatial maps.

Dense(128, relu) → small fully connected layer.

Dropout(0.5) → regularization.

Dense(4, softmax) → 4‑class output.

Compilation: Adam (1e‑4), categorical cross‑entropy, accuracy metric.

**7. Callbacks**
EarlyStopping — stop if val‑loss doesn’t improve for 5 epochs, restore best weights.

ReduceLROnPlateau — drop learning rate by ×0.2 if val‑loss plateaus for 2 epochs.

**8. Training**
Runs up to 30 epochs with the training generator, validates on the val generator, uses the class‑weight dictionary, and applies the two callbacks. Saves the best model to Project_1_Alzhiemers.keras.

**9. Evaluation**
Predictions on the test generator.

Classification report — precision, recall, F1 per class.

Confusion matrix — plotted as a heatmap for quick visual error analysis.

**10. Quick Sample Predictions**
Shows three random test images with their predicted vs. true labels for a qualitative sanity check.

**11. Reloading the Model**
Loads the saved .keras file so you can use it outside the training session. A fixed list class_names defines the human‑readable order of classes.

**12. Grad‑CAM Utilities**
preprocess_image — loads any image, resizes to 224×224, rescales, and adds a batch dimension.

generate_gradcam

Builds a mini‑model that outputs both the activation map of Conv_1 (last conv layer) and the final prediction.

Computes the gradient of the chosen class score relative to that activation map.

Weights each feature map by its average gradient, then collapses into a 2‑D heatmap, ReLU’s it, and normalizes to [0, 1].

display_gradcam — wraps everything:

Preprocess image → predict class → generate Grad‑CAM → upscale heatmap → apply OpenCV’s JET colormap → overlay at 40 % opacity on the original image → show with matplotlib.

**13. Grad‑CAM Demo Call**
Sets img_path to a specific MRI scan and runs display_gradcam(img_path), yielding a color overlay that highlights the brain regions most influential for the model’s Alzheimer stage prediction.

That’s the whole workflow: prepare data → build & fine‑tune a MobileNetV2 classifier → evaluate quantitatively & qualitatively → interpret predictions with Grad‑CAM.
