%% Read data
samples = load('data/heart_P.mat');
samples = samples.heart_P;
labels = load('data/heart_T.mat');
labels = labels.heart_T;

%% Manipulate input data so it's easier for network to understand
% normalize continuous input and dummyvar categorical

% Age
samples(1, :) = zscore(samples(1, :));

% Sex - adds 1 row
samples(2, :) = samples(2, :) + 1;
samples = cat_to_dummy(samples, 2);

% Chest Pain Type - adds 3
samples(4, :) = samples(4, :) + 1;
samples = cat_to_dummy(samples, 4);

% Resting blood pressure
samples(8, :) = zscore(samples(8, :));

% Cholesterol
samples(9, :) = zscore(samples(9, :));

% Fasting blood sugare > 120 adds 1 row
samples(10, :) = samples(10, :) + 1;
samples = cat_to_dummy(samples, 10);

% Resting EC Results - don't know if this is needed. Adds 2 rows
samples(12, :) = samples(12, :) + 1;
samples = cat_to_dummy(samples, 12);

% Max heart rate
samples(15, :) = zscore(samples(15, :));

% Exercise induced angina - adds 1 row
samples(16, :) = samples(16, :) + 1;
samples = cat_to_dummy(samples, 16);

% old peak
samples(18, :) = zscore(samples(18, :));

% Slope - adds 2 rows
samples(19, :) = samples(19, :) + 1;
samples = cat_to_dummy(samples, 19);

% Number of major vessels colored by fluoroscopy - adds 4 rows
samples(22, :) = samples(22, :) + 1;
samples = cat_to_dummy(samples, 22);

% Thal - adds 2 rows
samples(27, :) = samples(27, :) + 1;
samples = cat_to_dummy(samples, 27);

%% Set up network
train_fcn = 'trainlm';
hidden_layer_1_sz = 64;
hidden_layer_2_sz = 10;

net = fitnet([hidden_layer_1_sz, hidden_layer_2_sz], train_fcn);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% division of the inputdata is done automatically in this configuration
net.performFcn = 'mse';

%% Train Network
% [best_net, best_roc] = eval_mlp_network(net, samples, labels, 10);
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

save('models/task_3_v3.mat', 'net');