%% DATA PREPARATION
trim_scale = 0; %how many pixels to trim 
image_size = [28-trim_scale,28-trim_scale];
%to reshape the images we need to know their overall shape
image_vector_length = image_size(1)*image_size(2); 
%multiply the first and second index of the image_size to determine the total vector length
training_samples = 60000; %select the number of sample 
[images, labels]= readMNIST_vector('fashion_train-images-idx3-ubyte','fashion_train-labels-idx1-ubyte',training_samples,0,trim_scale); % initialize figure  
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

hiddenLayer1Size = 64; % total number of nodes in the hidden layer 1
hiddenLayer2Size = 32; % total number of nodes in the hidden layer 2

% create a FitNet Feedforward Network
net = patternnet([hiddenLayer1Size hiddenLayer2Size, 26],'trainrp');

% division of the inputdata is done automatically in this configuration
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 25/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'crossentropy';
net.trainParam.epochs = 1200;

[net,tr] = train(net,x_train,train_labels);

%use the train function to learn the lables of the images in the mnist data set
%we pass the 'net' structure we've built above along with the x_train(sample_array) and train-labels (source_labels^Transpose) 


%% DISPLAY DATA
% This is the functio we used in class to show the samples as images -> the labels are shown in decimal form (updated)  

figure(1)                                       % initialize figure
colormap(gray)                                  % set to grayscale
for i = 1:36                                    % preview first 36 samples
    subplot(6,6,i)                              % plot them in 6 x 6 grid
    digit = reshape(sample_array(:,i), image_size);     % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(labels(i)))                   % show the label above each cell of the plot
end

%% Display ROC scores
preds = sim(net, x_train);

figure(2)
plotroc(train_labels, preds);

[tpr, fpr, thresholds] = roc(train_labels, preds);
roc_auc_scores = zeros(1, 10);

for i = 1 : 10
    roc_auc_scores(1, i) = trapz(fpr{i}, tpr{i});
end

avg_auc = mean(roc_auc_scores);
disp(avg_auc);

figure(3)
title('ROC AUC per class');
xlabel('Class');
ylabel('ROC AUC');

bar(roc_auc_scores);

save('models\fashion_task_3_v3.mat', 'net');