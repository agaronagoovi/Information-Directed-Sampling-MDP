param = 1:10;
MDPs = cell(length(param),1);

for i=1:length(param)
    [MDPs{i,1}.ValueFunction,MDPs{i,1}.ScrapThreshold,MDPs{i,1}.minimumBid,MDPs{i,1}.Pssa] = valueIteration(param(i));
    
end

pd = makedist('NegativeBinomial','R',5,'p',0.5);
t = truncate(pd,1,length(param));
conjugatePrior = pdf(t,1:length(param))';
clear i;