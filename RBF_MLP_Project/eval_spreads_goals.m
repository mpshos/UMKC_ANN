function [best_net, trainMse, valMse] = eval_spreads_goals(trainP,trainT, valP, valT, spreads, goals)
%EVAL_SPREADS Create an exact RBF for each provided spread
% Initialize
trainMse = zeros(length(spreads));
valMse = zeros(length(spreads));
least_sum_mse = 100;

for i = 1 : length(spreads)
    for j = 1 : length(goals)
        
       % Create/train network
       net = newrb(trainP, trainT, goals(j), spreads(i));

        yv = sim(net, valP);
        valMse(i) = mse(yv - valT);

        ytr = sim(net, trainP);
        trainMse(i) = mse(ytr - trainT);

        if (valMse(i) + trainMse(i)) < least_sum_mse
            least_sum_mse = valMse(i) + trainMse(i);
            best_net = net;
        end
    end
end

end


