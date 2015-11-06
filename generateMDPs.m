param = 1:11;
MDPs = cell(length(param),2);

for i=1:length(param)
    [MDPs{i,1},MDPs{i,2}] = makeMDP(param(i));
end