function [ret_net, best_spread] = best_spread_rbe(train_sample, train_label, val_samples, val_labels, spreads)
%BEST_SPREAD_RBE Summary of this function goes here
%   Try each spread and return the best performer
best_spread = 0;
max_mse = 0;

for i = 1 : length(spreads)
    net = newrbe(train_sample, train_label, spreads(i));
    
    yv = sim(net, val_samples);
    msev = mse(yv - val_labels);
    
%     disp(spreads(i));
%     disp(msev);
    
    if msev > max_mse
        max_mse = msev;
        best_spread = spreads(i);
        ret_net = net;
    end
end
end

