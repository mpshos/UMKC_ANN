% CNN Training for EMNIST_LETTERS

letter_training_size = 124800; %The EMNIST_LETTERS dataset contains 145,600 samples, but only 124800 are part of the training set

[I,labels,I_test,labels_test] = emnist_readMNIST_cell(letter_training_size);%This function has been customized to import the files for letter data

I = I'; %transpose the samples
Train = table(I); %convert to a table
Train = [Train table(labels)];%concatinate the table with the lables 
Train.Properties.VariableNames{2} = 'Labels'; %change the variable name of the lable data
Train.Labels = categorical(Train.Labels); %convert the numerical lables to categorical labels (matlab internal requirment for classification layer)

split_ratio = 0.6;
%generate a training and validation split for the training data above
ranRows = randperm(letter_training_size);%randomly permute the labels of the training data
%for *repeatability* you can generate this only once if you prefer to use the same subset for any comparison
AlphaTrain = Train(ranRows(1:(split_ratio*letter_training_size)),:);%copy the training set
AlphaValidation = Train(ranRows((split_ratio*letter_training_size)+1:end),:);%copy the test set

%repeate the process above for the testing set
I_test= I_test';
Test = table(I_test);
Test = [Test table(labels_test)];
Test.Properties.VariableNames{2} = 'Labels';
Test.Labels = categorical(Test.Labels);

%generate the structure of the model we'll use
layers = [
    imageInputLayer([28 28 1]) %the array passed here needs to be the size of the input sample 28x28 and 1 means its a flat (2D) array
    %start block 1
    convolution2dLayer(7,8,'Padding','same')%create a 2D conv layer 
    batchNormalizationLayer %create a layer to reduce sample variance -- 'normalize' the updates for the batch 
    reluLayer %create a ReLU layer 
    
    maxPooling2dLayer(2,'Stride',2) %pooling reduces the size of the feature tensor you send to the next layer
    %end block 1

    %start block 2
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %end block 2
    
    %start block 3
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %end block 3
    
    fullyConnectedLayer(26)%connect flattened outputs of block 3 to an MLP or 'FC'NN layer
    softmaxLayer %categorical probability selection layer
    classificationLayer];%final classification layer -- generates the results

%/Add unique names to the layers above(required for layerGraph generation)
layers(1).Name = 'Input';
layers(2).Name = 'Conv_1';
layers(3).Name = 'Bach_Norm_1';
layers(4).Name = 'ReLu_1';
layers(5).Name = 'Pool_1';
layers(6).Name = 'Conv_2';
layers(7).Name = 'Bach_Norm_2';
layers(8).Name = 'ReLu_2';
layers(9).Name = 'Pool_2';
layers(10).Name = 'Conv_3';
layers(11).Name = 'Bach_Norm_3';
layers(12).Name = 'ReLu_3';
layers(13).Name = 'Fully_Connected';
layers(14).Name = 'Soft_Max';
layers(15).Name = 'Class_Out';

analyzeNetwork(layers)%generate a visual of the model for inspection

%tell the training system what parameters our model will use
options = trainingOptions('adam', ... %change the learing method
    'InitialLearnRate',0.01, ... %base learning rate
    'LearnRateSchedule','piecewise', ... %allows the learning rate to be reducecd on a schedule (start fast and slow down for more refined (smaller) steps)
    'LearnRateDropPeriod',1, ... %how many epocs before learning rate reduces 
    'LearnRateDropFactor',0.002, ... %how much to reduce the learning rate (small number) 
    'MiniBatchSize', 256, ... %how many samples per batch
    'ValidationPatience',5, ... %(Early stopping)number of times that the loss on the validation set can be larger than or equal to the previously smallest loss before network training stops
    'L2Regularization',0.0005, ... %weight decay factor (large numbers will increase the necessary training time
    'GradientDecayFactor',0.95, ... %gradient decay of the ADAM solver   
    'SquaredGradientDecayFactor',0.99, ... %squared decay of the ADAM solver
    'MaxEpochs',7, ... %how long to train the whole model -- longer could mean better results, but also means much longer training time
    'Shuffle','every-epoch', ... %reshuffle the batches every time we cycle through the data set
    'ValidationData',AlphaValidation, ... %use validation data supplied by the split we did above
    'ValidationFrequency',25, ... %how offten do we want to check the model and make sure it isn't momorizing
    'Verbose',false, ... %this can allow you to log the training in the command window -- it is suppressed here to save resources
    'Plots','training-progress'); %you can turn this off if you want to train without the visual feedback -- turn on verbose and watch the logs if you run into resourse issues

trainedNet = trainNetwork(AlphaTrain,layers,options);%start the training by calling the data table and layers / options objects that describe the model

%Test the network's performance on unseen data

YPred = classify(trainedNet,Test);
YValidation = Test.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

save('models/task_2_v3.mat', 'trainedNet');
