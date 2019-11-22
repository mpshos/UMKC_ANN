%% Read data
samples = load('data/P.mat');
P = samples.P;
% P = normr(samples.P);
labels = load('data/T.mat');

% Convert labels to boolean
T = (labels.T + 1) / 2;

[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,0.6,0.2,0.2);
[trainT,valT,testT] = divideind(T,trainInd,valInd,testInd);

% Normalize data
% samples_nrm = normc(samples);

%% Set up networks
net1 = newrbe(trainP,trainT,3.0);

y1v = sim(net1,valP);
mse1v=mse(y1v-valT);

[net, spread] = best_spread_rbe(trainP, trainT, valP, valT, [0.05, 0.1, 0.25, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

preds = sim(net, testP);

figure(1)
plotroc(testT, testP);

% Use threshold to determine class
preds(preds >= 0.5) = 1;
preds(preds < 0.5) = 0;

% Calculate and plot ROC AUC
[X, Y, T, AUC] = perfcurve(valT, preds, 1);

disp(AUC);

figure(2);
confusionchart(valT, preds);



