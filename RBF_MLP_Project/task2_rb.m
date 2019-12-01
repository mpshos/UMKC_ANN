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
% net1 = newrb(trainP,trainT, 10.0, 10);
% 
% y1v = sim(net1,valP);
% mse1v=mse(y1v-valT);
% disp(mse1v);


spreads = 0.5 : 0.5 : 10;
goals = 0.0 : 0.1 : 1.9;

[net, trainMse, valMse] = eval_spreads_goals(trainP, trainT, valP, valT, spreads, goals);

figure(1)
plot3(spreads, goals, trainMse);
title('Train MSE vs Spread');

figure(2)
plot3(spreads, goals, valMse);
title('Validation MSE vs Spread');


% preds = sim(net, testP);

% figure(
% plotroc(testT, testP);
% 
% % Use threshold to determine class
% preds(preds >= 0.5) = 1;
% preds(preds < 0.5) = 0;
% 
% % Calculate and plot ROC AUC
% [X, Y, T, AUC] = perfcurve(valT, preds, 1);
% 
% disp(AUC);
% 
% figure(2);
% confusionchart(valT, preds);



