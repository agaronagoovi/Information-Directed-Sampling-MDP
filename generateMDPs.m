param = 5:5:50;
MDPs = cell(length(param),2);

for i=1:length(param)
    [MDPs{i,1},MDPs{i,2}] = mdpDoseResponse(param(i));
end

initQ = zeros(size(MDPs{i,1},1),size(MDPs{i,1},3));
pd = makedist('NegativeBinomial','R',5,'p',0.5);
t = truncate(pd,1,length(param));
conjugatePrior = pdf(t,1:length(param))';
clear i;