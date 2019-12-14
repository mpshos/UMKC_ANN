function [best_net,best_roc] = eval_mlp_network(net,samples, labels, iterations)
%EVAL_MLP_NETWORK Summary of this function goes here
%   Detailed explanation goes here
best_roc = 0;
best_net = net;

for i = 1 : iterations
    init(net);
    
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
    
    if AUC > best_roc
        best_roc = AUC;
        best_net = net;
    end
end
end

