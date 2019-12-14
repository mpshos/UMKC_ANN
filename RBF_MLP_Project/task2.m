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

%% Train a bunch of RBFs

spreads = 0.05 : 0.1 : 50;

[net, best_spread, trainMse, valMse] = eval_spreads(trainP, trainT, valP, valT, spreads);

figure(1)
plot(spreads, trainMse);
title('Train MSE vs Spread');

figure(2)
plot(spreads, valMse);
title('Validation MSE vs Spread');


preds = sim(net, testP);

figure(3)
plotroc(testT, testP);

% Use threshold to determine class
preds(preds >= 0.5) = 1;
preds(preds < 0.5) = 0;

% Calculate and plot ROC AUC
[X, Y, T, AUC] = perfcurve(valT, preds, 1);

disp(AUC);
disp(best_spread);

figure(4);
confusionchart(valT, preds);



