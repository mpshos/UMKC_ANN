% CNN Training for FASHION_MNIST

training_size = 60000; %The FASHION_MNIST data set size
numClasses = 10;
[F,F_labels,F_test,F_labels_test] = fashion_readMNIST_cell(training_size);%This function has been customized to import the files for letter data

F = F'; %transpose the samples
Fashion_Train = table(F); %convert to a table
Fashion_Train = [Fashion_Train table(F_labels)];%concatinate the table with the lables 
Fashion_Train.Properties.VariableNames{2} = 'Labels'; %change the variable name of the lable data
Fashion_Train.Labels = categorical(Fashion_Train.Labels); %convert the numerical lables to categorical labels (matlab internal requirment for classification layer)

split_ratio = 0.6;
%generate a training and validation split for the training data above
ranRows = randperm(training_size);%randomly permute the labels of the training data
%for *repeatability* you can generate this only once if you prefer to use the same subset for any comparison
Clothes_Train = Fashion_Train(ranRows(1:(split_ratio*training_size)),:);%copy the training set
Clothes_Validation = Fashion_Train(ranRows((split_ratio*training_size)+1:end),:);%copy the test set

%repeate the process above for the testing set
F_test= F_test';
Clothes_Test = table(F_test);
Clothes_Test = [Clothes_Test table(F_labels_test)];
Clothes_Test.Properties.VariableNames{2} = 'Labels';
Clothes_Test.Labels = categorical(Clothes_Test.Labels);

%generate the structure of the model we'll use
layers = [
    imageInputLayer([28 28 1]) %the array passed here needs to be the size of the input sample 28x28 and 1 means its a flat (2D) array
    %start block 1
    convolution2dLayer(3,8,'Padding','same')%create a 2D conv layer 
    batchNormalizationLayer %create a layer to reduce sample variance -- 'normalize' the updates for the batch 
    reluLayer %create a ReLU layer 
    
    maxPooling2dLayer(2,'Stride',2) %pooling reduces the size of the feature tensor you send to the next layer
    %end block 1

    %start block 2
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %end block 2
    
    %start block 3
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    %end block 3
    
    fullyConnectedLayer(numClasses)%connect flattened outputs of block 3 to an MLP or 'FC'NN layer
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
options = trainingOptions('adam', ... %training function
    'InitialLearnRate',0.01, ... %initial learing rate can be modified by using an update *schedule*(slow the learning rate after some number of epocs)
    'MaxEpochs',4, ... %how long to train the whole model -- longer could mean better results, but also means much longer training time
    'Shuffle','every-epoch', ... %reshuffle the batches every time we cycle through the data set
    'ValidationData',Clothes_Validation, ... %use validation data supplied by the split we did above
    'ValidationFrequency',25, ... %how offten do we want to check the model and make sure it isn't momorizing
    'Verbose',false, ... %this can allow you to log the training in the command window -- it is suppressed here to save resources
    'Plots','training-progress'); %you can turn this off if you want to train without the visual feedback -- turn on verbose and watch the logs if you run into resourse issues

trainedNet = trainNetwork(Clothes_Train,layers,options);%start the training by calling the data table and layers / options objects that describe the model

%Test the network's performance on unseen data

YPred = classify(trainedNet,Clothes_Test);
YValidation = Clothes_Test.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

save('models/fashion_task_1_v2.mat', 'trainedNet');
