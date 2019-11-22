%% Read data
samples = load('data/heart_P.mat');
samples = samples.heart_P;
labels = load('data/heart_T.mat');
labels = labels.heart_T;

% Normalize data
samples_nrm = normr(samples);

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