%% Transfer Learning %%

% LOADS THE EMNIST DATA AND TESTS WHATEVER MODEL IS IN THE WORKSPACE

%********This code must be run -after- a CNN training!*********************

% The folling elements will remove the final layers of the model and insert new 
% ones which can be use to fine tune a model to new data, or test the model on 
% different data with different labels

% Load best Task 3 network
trainedNet = load('models/task_3_v3.mat').trainedNet;

%% EMNIST_LETTERS DATASET IMPORT
letter_training_size = 124800; %The EMNIST_LETTERS dataset contains 145,600 samples, but only 124800 are part of the training set

[Letters,Letters_labels,Letters_test,Letters_labels_test] = emnist_readMNIST_cell(letter_training_size);%This function has been customized to import the files for letter data

Letters = Letters'; %transpose the samples
Letters_Train = table(Letters); %convert to a table
Letters_Train = [Letters_Train table(Letters_labels)];%concatinate the table with the lables 
Letters_Train.Properties.VariableNames{2} = 'Labels'; %change the variable name of the lable data
Letters_Train.Labels = categorical(Letters_Train.Labels); %convert the numerical lables to categorical labels (matlab internal requirment for classification layer)

split_ratio = 0.8;
%generate a training and validation split for the training data above
ranRows = randperm(letter_training_size);%randomly permute the labels of the training data
%for *repeatability* you can generate this only once if you prefer to use the same subset for any comparison
New_Train = Letters_Train(ranRows(1:(split_ratio*letter_training_size)),:);%copy the training set
New_Validation = Letters_Train(ranRows((split_ratio*letter_training_size)+1:end),:);%copy the test set

%repeate the process above for the testing set
Letters_test= Letters_test';
New_Test = table(Letters_test);
New_Test = [New_Test table(Letters_labels_test)];
New_Test.Properties.VariableNames{2} = 'Labels';
New_Test.Labels = categorical(New_Test.Labels);


%%MAKE A COPY OF THE NETWORK%%

new_trainedNet = trainedNet;

%% Generate a New Layer Graph and Update the - FC / SM / CLASS - Layers

numClasses = numel(categories(New_Train.Labels));%Train here is the NEW data
lgraph = layerGraph(layers);%Make a copy of the layers using the layerGraph function

%Make changes to layers by making a new layer and replacing the old one in
%the graph
%make a new fully connected layer
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fullyconnected',...
                      'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);

%replace the Fully_Connected layer in the trained network with the new one
lgraph = replaceLayer(lgraph,'Fully_Connected',newFCLayer);

%make a new softmax and replace the old one
newSoftMax = softmaxLayer('Name','new_softmax');
lgraph = replaceLayer(lgraph,'Soft_Max',newSoftMax);

%make a new classification layer and replace the old one
newClassLayer = classificationLayer('Name','new_classout');
lgraph = replaceLayer(lgraph,'Class_Out',newClassLayer);

%you can select new training options as well by creating a new options
%object -- added as comments are some things you can do *if you want* -- they are not required and only a guideline

alternative_options = trainingOptions('adam', ... %parameters for the ADAM learing method
    'InitialLearnRate',0.01, ... %select a learning rate
    'LearnRateSchedule','piecewise', ... %allows the learning rate to be reducecd on a schedule (start fast and slow down for more refined (smaller) steps)
    'LearnRateDropPeriod',1, ... %how many epocs before learning rate reduces 
    'LearnRateDropFactor',0.002, ... %how much to reduce the learning rate (small number) 
    'MiniBatchSize', 64, ... %how many samples per batch
    'ValidationPatience',5, ... %(Early stopping)number of times that the loss on the validation set can be larger than or equal to the previously smallest loss before network training stops
    'L2Regularization',0.0005, ... %weight decay factor (large numbers will increase the necessary training time
    'GradientDecayFactor',0.95, ... %gradient decay of the ADAM solver   
    'SquaredGradientDecayFactor',0.99, ... %squared decay of the ADAM solver
    'MaxEpochs', 3, ... %select how many epocs you think will be necessary for 92% or greater accuracy
    'Shuffle','every-epoch', ... 
    'ValidationData',New_Validation, ... %be SURE to update the validation data for your new data set
    'ValidationFrequency',50, ... %increase the validation frequency
    'Verbose',false, ... 
    'Plots','training-progress'); 

%retrain the network
new_trainedNet = trainNetwork(New_Train,lgraph,alternative_options);%use the updated layer graph (lgraph) and the new options to train the new data


% Evaluate the Accuracy of the New_Test data
YPred = classify(new_trainedNet,New_Test);
YValidation = New_Test.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

save('models/task_4_v1.mat', 'trainedNet');
