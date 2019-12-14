function [best_net, best_spread, trainMse, valMse] = eval_spreads(trainP,trainT, valP, valT, spreads)
%EVAL_SPREADS Create an exact RBF for each provided spread
% Initialize
best_spread = 1000;
trainMse = zeros(length(spreads));
valMse = zeros(length(spreads));
least_sum_mse = 100;

for i = 1 : length(spreads)
   % Create/train network
   net = newrbe(trainP, trainT, spreads(i));
    
    yv = sim(net, valP);
    valMse(i) = mse(yv - valT);
    
    ytr = sim(net, trainP);
    trainMse(i) = mse(ytr - trainT);
    
    if (valMse(i) + trainMse(i)) < least_sum_mse
        least_sum_mse = valMse(i) + trainMse(i);
        best_net = net;
        best_spread = spreads(i);
    end
   
end

end


