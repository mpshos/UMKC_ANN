%% DATA PREPARATION
% 
trim_scale = 0; %how many pixels to trim 
image_size = [28-(2 * trim_scale),28-(2 *trim_scale)];
%to reshape the images we need to know their overall shape
image_vector_length = image_size(1)*image_size(2); 
%multiply the first and second index of the image_size to determine the total vector length

training_samples = 60000; %select the number of sample 

[images, labels]= readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',training_samples,0,trim_scale); % initialize figure  
% see the function source readMNIST.m for more info on the data set reading function
% be sure to include the file in your working directory, or add the directory to path

source_labels = labels';                                    
% transpose to match the format of the 'dummyvar' function
% we need to use this function to build the 'label' vector for each sample
% [labels] contains decimal values indicating the digit assocated with each sample
% converting that value to a 'one-hot binary' notation is necessary for calculations of our performFcn
source_labels(source_labels==0)=10;                         
% 'dummyvar' function doesn´t take zeroes so we *shift* the '0' label to '10'
% the resulting vector will be indexed from 1 to 0 -> [1 2 3 4 5 6 7 8 9 0]
source_labels=dummyvar(source_labels); 
% use dummyvar to convert the [scalar] decmial label to the one-hot [vector] array

sample_array = reshape(images,image_vector_length,training_samples);
%we need to vectorize the images read in from the readMNIST function, dimensions need to be maintained

%% TRAIN NETWORK
x_train = sample_array;        
% it is conventional to copy the data to a new array (x) once you've got it in the correct shape
train_labels = source_labels'; 
% addign the name 'train' to your variables can make it easier to determine which matrix you're working with once you start including a test set

hiddenLayerSize = 70;                          % total number of nodes in the hidden layer
net = patternnet(hiddenLayerSize, 'trainscg');             % create Pattern Recognition Network

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% division of the inputdata is done automatically in this configuration
net.performFcn = 'crossentropy';
% in order to propigate error we need to determine 

[net,tr] = train(net,x_train,train_labels);
%use the train function to learn the lables of the images in the mnist data set
%we pass the 'net' structure we've built above along with the x_train(sample_array) and train-labels (source_labels^Transpose) 

%% LOG TRAINING PERFORMANCE
test_case_output = '1024.mat';
save(test_case_output, 'tr');
disp(tr)

%% DISPLAY DATA
% This is the functio we used in class to show the samples as images -> the labels are shown in decimal form (updated)  

figure                                          % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:36                                    % preview first 36 samples
    subplot(6,6,i)                              % plot them in 6 x 6 grid
    digit = reshape(sample_array(:,i), image_size);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(labels(i)))                   % show the label above each cell of the plot
end