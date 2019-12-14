%% Read data
samples = load('data/heart_P.mat');
samples = samples.heart_P;
labels = load('data/heart_T.mat');
labels = labels.heart_T;

% Normalize data
% samples_nrm = normr(samples);

% Convert categorical data to dummy var
% Sex - adds 1 row
samples(2, :) = samples(2, :) + 1;
samples = cat_to_dummy(samples, 2);

% Chest Pain Type - adds 3
samples(4, :) = samples(4, :) + 1;
samples = cat_to_dummy(samples, 4);

% Fasting blood sugare > 120 adds 1 row
samples(10, :) = samples(10, :) + 1;
samples = cat_to_dummy(samples, 10);

% Resting EC Results - don't know if this is needed. Adds 2 rows
samples(12, :) = samples(12, :) + 1;
samples = cat_to_dummy(samples, 12);

% Exercise induced angina - adds 1 row
samples(16, :) = samples(16, :) + 1;
samples = cat_to_dummy(samples, 16);

% Number of major vessels colored by fluoroscopy - adds 4 rows
samples(20, :) = samples(20, :) + 1;
samples = cat_to_dummy(samples, 20);

% Thal - adds 2 rows
samples(25, :) = samples(25, :) + 1;
samples = cat_to_dummy(samples, 25);

%% Set up network
train_fcn = 'trainlm';
hidden_layer_1_sz = 50;
hidden_layer_2_sz = 10;

net = fitnet([hidden_layer_1_sz, hidden_layer_2_sz], train_fcn);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% division of the inputdata is done automatically in this configuration
net.performFcn = 'mse';

%% Train Network
[net,tr] = train(net,samples,labels);

%% Calculate performance
% Run all samples through network and get output
preds = sim(net, samples);

% Use threshold to determine class
preds(preds >= 0.5) = 1;
preds(preds < 0.5) = 0;

% Calculate and plot ROC AUC
[X, Y, T, AUC] = perfcurve(labels, preds, 1);

disp(AUC);

figure(1);
plot(X, Y);

figure(2);
confusionchart(labels, preds);

save('models/task_3_v2.mat', 'net');