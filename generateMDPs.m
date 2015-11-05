params = 0:0.2:2;
MDPs = cell(length(params),2);

for i=1:length(params)
    [MDPs{params,1},MDPs{params,2}] = makeMDP(params(i));
end