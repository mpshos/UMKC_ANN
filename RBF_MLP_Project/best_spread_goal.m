function [spread,goal] = best_spread_goal(train_sample, train_label, val_samples, val_labels, spreads, goals)
%BEST_SPREAD_GOAL Summary of this function goes here
%   Detailed explanation goes here
spread = 0
goal = 0

for i = 1: length(spread)
   for j=1 : length(goal)
      net = newrb(train_sample,train_label,goals(j),spreads(i)); 
      
      y2tr= sim(net,train);
      mse2tr=mse(y2tr-trainT);
   end
end
end

