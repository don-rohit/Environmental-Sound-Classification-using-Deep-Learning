Deep Learning Problem Statement
Supriya Kumari

Data Pre-processing

Given Dataset Contain 2000 environmental audio (.wav) recordings. The dataset consists of 5 seconds recordings organized into 50 semantical classes (with 40 examples per class) and arranged into 5 major categories.

 

File name Description:

Example file name: 1-137-A-32.wav

{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
•	{FOLD} - index of the cross-validation fold,
•	{CLIP_ID} - ID of the original Freesound clip,
•	{TAKE} - letter disambiguating between different fragments from the same Freesound clip,
•	{TARGET} - class in numeric format [0, 49]
Listing all environmental sound audio file using os library of python and extracted all file target class in a list labels.
  
The dataset consists of 5 seconds recordings converted it into 10 seconds recording for better classification.
Using AudioSegment from pydub library of python.
Loaded File using AudioSegment.from_wav(Path_of_files)  and replicate it to make 10 second long and export into same path using export  function
 

After that it will convert all 2000 environmental sound audio to 10 sec and listed all labels in a python list.
As 2000 data is very less to classify 50 classes did Data Augmentation by varying pitch of sound as we can’t vary time as we have to deal on 10 sec and can’t make remix as it is monotone audio file.
Pitch is varied using librosa.effects.pitch_shift() on 2000 data and export it.
y, sr = librosa.load(file_path[i])  
y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
where n_steps = 2 #-1, -2, 2, 1 and n_steps = 2.5 #-1, -2, 2, 1
and now we have 6000 data! Then implemented 10 fold cross validation using sklearn  python library.
(from sklearn.model_selection import StratifiedKFold) and then splited the dataset and reshaped it and did one hot encoding to pass it to the model.
 


Spectrum Generation

A spectrogram is a visual representation of the spectrum of frequencies of sound or other signal as they vary with time.
Spectrogram can be generated using FFT (Fast Fourier Transform) or librosa library of Python
Converting Audio file in Spectrogram:
I used librosa for converting audio file into spectrogram. 
Used librosa.load() to load a audio file as a floating point time series and used time series 2.97, librosa.feature.melspectrogram() to Compute a mel-scaled spectrogram, librosa.amplitude_to_db() Convert an amplitude spectrogram to dB-scaled spectrogram and flatten it for classification further.

 





Waveform of a Audio File 
 
Spectrogram of a Audio File

 
 

Model Preparation 

Convolutional Neural Network with 5 layers
 
 
I have used 10 fold cross validation to train the model showing last one with accuracy 0f 82%
 

Result

My model is simple one using just 5 layers which takes not much time to train as well it uses no fancy techniques thus easy to understand as well easy to train and can be transformed easily to adopt any other technique above it, for example at first I trained the model on just 2000 data  and got a test accuracy of 54% then added data augmentation and 10 fold cross validation on top of it to achieve the accuracy achieved was 82%.
I think I have trained the model correctly as it predicts decently on unseen data as well on real time data.



