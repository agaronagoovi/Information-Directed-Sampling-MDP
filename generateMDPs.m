param = 1:2:20;
MDPs = cell(length(param),2);

for i=1:length(param)
    [MDPs{i,1},MDPs{i,2}] = mdpAuctionLotSizing(param(i));
end

initQ = zeros(size(MDPs{i,1},1),size(MDPs{i,1},3));
pd = makedist('NegativeBinomial','R',5,'p',0.5);
t = truncate(pd,1,length(param));
conjugatePrior = pdf(t,1:length(param))';
MDPs = generateValueFunctions( MDPs,param );
clear i;