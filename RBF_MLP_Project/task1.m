%% Read data
samples = load('data\P.mat');
samples = samples.P;
labels = load('data\T.mat');

% Convert labels to boolean
labels = (labels.T + 1) / 2;

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
[net,tr] = train(net,samples_nrm,labels);

%% Calculate performance
% Run all samples through network and get output
preds = sim(net, samples_nrm);

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

save('models\task_1_v2.mat', 'net');