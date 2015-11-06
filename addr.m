function i = addr(x1,x2,x3,mqlength)
%
%

mqlength = mqlength+1;
i = x3 + (x2-1+(x1-1)*mqlength)*mqlength; 

end