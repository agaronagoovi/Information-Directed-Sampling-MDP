function [ Pssa, L] = toyMDP( p )
%TOYMDP Summary of this function goes here
%   Detailed explanation goes here

Pssa = zeros(2,2,2);
Pssa(1,1,1) = 1-p;
Pssa(1,2,1) = p;
Pssa(1,1,2) = 0.1;
Pssa(1,2,2)=0.9;
Pssa(2,1,1)=0.3;
Pssa(2,2,1)=0.7;
Pssa(2,1,2)=0.8;
Pssa(2,2,2)=0.2;

L(1) = 1;
L(2) = 10;

end

