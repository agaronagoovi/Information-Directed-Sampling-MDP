param = 5:5:50;

MDPs = cell(length(param),1);

for i=1:length(param)
    [MDPs{i,1}.Pssa,MDPs{i,1}.Rssa,MDPs{i,1}.h,N] = mdpDoseResponse_finite(param(i));
    [MDPs{i,1}.ValueFunction,MDPs{i,1}.policy,~] = mdp_finite_horizon(MDPs{i,1}.Pssa,MDPs{i,1}.Rssa, 1, N, MDPs{i,1}.h);
    MDPs{i,1}.param = param(i);
end

initQ = zeros(size(MDPs{i,1}.Pssa,1),size(MDPs{i,1}.Pssa,3));
pd = makedist('NegativeBinomial','R',5,'p',0.5);
t = truncate(pd,1,length(param));
conjugatePrior = pdf(t,1:length(param))';

clear i;

