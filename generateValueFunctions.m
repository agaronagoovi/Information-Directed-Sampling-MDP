function [ valueMDPs ] = generateValueFunctions( MDPs,param )
%GENERATEVALUEFUNCTIONS Summary of this function goes here
%   Detailed explanation goes here
valueMDPs = cell(length(param),1);
for i=1:length(param)
    valueMDPs{i,1}.ValueFunction = valueIteration(MDPs{i,1},MDPs{i,2});
    valueMDPs{i,1}.Pssa = MDPs{i,1};
    valueMDPs{i,1}.Rssa = MDPs{i,2};
    valueMDPs{i,1}.param = param(i);
end
end

