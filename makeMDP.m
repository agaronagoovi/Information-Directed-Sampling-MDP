function [ Pssa, L ] = makeMDP( lParam)
%MAKEMDP Summary of this function goes here
%   Detailed explanation goes here
mqlength = 5;
nu = 3;
nx = (mqlength+1)^3;
global l1 l2 m1 m2 m3;
l1 = 1;
l2 = 1;
m1 = 2;
m2 = lParam;
m3 = 3;
c = [-1;-1;-10];
global m_rate;
m_rate = l1+l2+m1+m2+m3;
alpha = 0.98;

Pssa = zeros(nx,nx,nu);
L = zeros(nx,1);

for q1=0:mqlength
    for q2=0:mqlength
        for q3=0:mqlength
            u = 1;
            pself = 0;
            p_q1next = 0;
            p_q1previous = 0;
            p_q2next = 0;
            p_q2previous = 0;
            p_q3next = 0;
            p_q3previous = 0;
            if q2~=mqlength
                if q1==0 && q3==0
                    rate = l1 + l2;
                    p_q1next = l1/(l1+l2);
                    p_q2next = l2/(l1+l2);
                elseif q1==0 && q3~=0
                    rate = l1 + l2 + m3;
                    p_q1next = l1/(l1+l2+m3);
                    p_q2next = l2/(l1+l2+m3);
                    p_q3previous = m3/(l1+l2+m3);
                elseif q1~=0 && q3==0
                    if q1~=mqlength
                        rate = l1 + m1 + l2;
                        p_q1next = l1/(l1+m1+l2);
                        p_q1previous = m1/(l1+m1+l2);
                        p_q2next = l2/(l1+m1+l2);
                    else
                        rate = m1 + l2;
                        p_q1previous = m1/(m1+l2);
                        p_q2next = l2/(m1+l2);
                    end
                else
                    if q1~=mqlength
                        rate = l1 + m1 + l2 + m3;
                        p_q1next = l1/(l1+m1+l2+m3);
                        p_q1previous = m1/(l1+m1+l2+m3);
                        p_q2next = l2/(l1+m1+l2+m3);
                        p_q3previous = m3/(l1+m1+l2+m3);
                    else
                        rate = m1 + l2 + m3;
                        p_q1previous = m1/(m1+l2+m3);
                        p_q2next = l2/(m1+l2+m3);
                        p_q3previous = m3/(m1+l2+m3);
                    end
                end
            else
                if q1==0 && q3==0
                    rate = l1;
                    p_q1next = 1;
                elseif q1==0 && q3~=0
                    rate = l1 + m3;
                    p_q1next = l1/(l1+m3);
                    p_q3previous = m3/(l1+m3);
                elseif q1~=0 && q3==0
                    if q1~=mqlength
                        rate = l1 + m1;
                        p_q1next = l1/(l1+m1);
                        p_q1previous = m1/(l1+m1);
                    else
                        rate = m1;
                        p_q1previous = 1;
                    end
                else
                    if q1~=mqlength
                        rate = l1 + m1 + m3;
                        p_q1next = l1/(l1+m1+m3);
                        p_q1previous = m1/(l1+m1+m3);
                        p_q3previous = m3/(l1+m1+m3);
                    else
                        rate =  m1 + m3;
                        p_q1previous = m1/(m1+m3);
                        p_q3previous = m3/(m1+m3);
                    end
                end
            end
            
            x1=q1+1;
            x2=q2+1;
            x3=q3+1;
            
            s = addr(x1,x2,x3,mqlength);
            if x1~=mqlength+1
                ns = addr(x1+1,x2,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q1next,pself);
            end
            if x1~=1
                ns = addr(x1-1,x2,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q1previous,pself);
            end
            
            if x2~=mqlength+1
                ns = addr(x1,x2+1,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q2next,pself);
            end
            
            if x3~=1
                ns = addr(x1,x2,x3-1,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q3previous,pself);
            end
            Pssa(s,s,u) = uniformizationself(rate,pself);
            
            
            u = 2;
            pself = 0;
            p_q1next = 0;
            p_q1previous = 0;
            p_q2next = 0;
            p_q2previous = 0;
            p_q3next = 0;
            p_q3previous = 0;
            if q1~=mqlength
                if q2==0 && q3==0
                    rate = l1 + l2;
                    p_q1next = l1/(l1+l2);
                    p_q2next = l2/(l1+l2);
                elseif q2==0 && q3~=0
                    rate = l1 + l2 + m3;
                    p_q1next = l1/(l1+l2+m3);
                    p_q2next = l2/(l1+l2+m3);
                    p_q3previous = m3/(l1+l2+m3);
                elseif q2~=0 && q3==0                       %constraint q2 previous and q3 next
                    if q2~=mqlength
                        if q3~=mqlength
                            rate = l1 + l2 + m2;
                            p_q1next = l1/(l1+l2+m2);
                            p_q2next = l2/(l1+l2+m2);
                            p_q2previous = m2/(l1+l2+m2);
                            p_q3next=1;
                        else
                            rate = l1 + l2;
                            p_q1next = l1/(l1+l2);
                            p_q2next = l2/(l1+l2);
                        end
                    else
                        if q3~=mqlength
                            rate = l1+ m2;
                            p_q1next = l1/(l1+m2);
                            p_q2next = l2/(l1+m2);
                            p_q2previous = m2/(l1+m2);
                            p_q3next=1;
                        else
                            rate = l1;
                            p_q1next = 1;
                        end
                    end
                else
                    if q2~=mqlength
                        if q3~=mqlength
                            rate = l1 + l2 + m2 + m3;
                            p_q1next = l1/(l1+l2+m2+m3);
                            p_q2next = l2/(l1+l2+m2+m3);
                            p_q2previous = m2/(l1+l2+m2+m3);
                            p_q3next=1;
                        else
                            rate = l1 + l2 + m3;
                            p_q1next = l1/(l1+l2+m3);
                            p_q2next = l2/(l1+l2+m3);
                        end
                        p_q3previous = m3/(l1+l2+m2+m3);
                    else
                        if q3~=mqlength
                            rate = l1 + m2 + m3;
                            p_q1next = l1/(l1+m2+m3);
                            p_q2previous = m2/(l1+m2+m3);
                            p_q3next=1;
                        else
                            rate = l1 + m3;
                            p_q1next = l1/(l1+m3);
                        end
                        p_q3previous = m3/(l1+m2+m3);
                    end
                end
            else
                if q2==0 && q3==0
                    rate = l2;
                    p_q2next = 1;
                elseif q2==0 && q3~=0
                    rate = l2 + m3;
                    p_q2next = l2/(l2+m3);
                    p_q3previous = m3/(l2+m3);
                elseif q2~=0 && q3==0
                    if q2~=mqlength
                        if q3~=mqlength
                            rate = l2 + m2;
                            p_q2next = l2/(l2+m2);
                            p_q2previous = m2/(l2+m2);
                            p_q3next=1;
                        else
                            rate = l2;
                            p_q2next = 1;
                        end
                    else
                        if q3~=mqlength
                            rate = m2;
                            p_q2previous = 1;
                            p_q3next=1;
                        else
                            rate = 0;
                        end
                    end
                else
                    if q2~=mqlength
                        if q3~=mqlength
                            rate = l2 + m2 + m3;
                            p_q2next = l2/(l2+m2+m3);
                            p_q2previous = m2/(l2+m2+m3);
                            p_q3next=1;
                        else
                            rate = l2 + m3;
                            p_q2next = l2/(l2+m3);
                        end
                        p_q3previous = m3/(l2+m2+m3);
                    else
                        if q3~=mqlength
                            rate = m2 + m3;
                            p_q2previous = m2/(m2+m3);
                            p_q3next=1;
                        else
                            rate = m3;
                        end
                        p_q3previous = m3/(m2+m3);
                    end
                end
            end
            
            x1=q1+1;
            x2=q2+1;
            x3=q3+1;
            
            s = addr(x1,x2,x3,mqlength);
            if x1~=mqlength+1
                ns = addr(x1+1,x2,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q1next,pself);
            end
            
            if x2~=mqlength+1
                ns = addr(x1,x2+1,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q2next,pself);
            end
%             if x2~=1 && x3~=mqlength+1
%                 ns = addr(x1,x2-1,x3,mqlength);
%                 [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q2previous*p_q3next,pself);
%             end
            
            if x3~=mqlength+1 && x2~=1
                ns = addr(x1,x2-1,x3+1,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q3next*p_q2previous,pself);
%             elseif x3==mqlength+1 && x2~=1
%                 ns = addr(x1,x2,x3,mqlength);
%                 [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q2previous,pself);
            end
            if x3~=1
                ns = addr(x1,x2,x3-1,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q3previous,pself);
            end
            
            Pssa(s,s,u) = uniformizationself(rate,pself);
            
            u=3;
            pself = 0;
            p_q1next = 0;
            p_q1previous = 0;
            p_q2next = 0;
            p_q2previous = 0;
            p_q3next = 0;
            p_q3previous = 0;
            
            if q1~=mqlength && q2==mqlength
                if q3~=0
                    rate = l1+m3;
                    p_q1next=l1/(l1+m3);
                    p_q3previous=m3/(l1+m3);
                else
                    rate = l1;
                    p_q1next = 1;
                end
            elseif q1==mqlength && q2~=mqlength
                if q3~=0
                    rate = l2+m3;
                    p_q2next=l2/(l2+m3);
                    p_q3previous=m3/(l2+m3);
                else
                    rate = l2;
                    p_q2next = 1;
                end
            elseif q1==mqlength && q2==mqlength
                if q3~=0
                    rate = m3;
                    p_q3previous=1;
                else
                    rate = 0;
                end
            else
                if q3~=0
                    rate = l1+l2+m3;
                    p_q1next=l1/(l1+l2+m3);
                    p_q2next=l2/(l2+l2+m3);
                    p_q3previous=m3/(l1+l2+m3);
                else
                    rate = l1+l2;
                    p_q1next = l1/(l1+l2);
                    p_q2next = l2/(l1+l2);
                end
            end
            
            x1=q1+1;
            x2=q2+1;
            x3=q3+1;
            
            s = addr(x1,x2,x3,mqlength);
            if x1~=mqlength+1
                ns = addr(x1+1,x2,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q1next,pself);
            end
            
            if x2~=mqlength+1
                ns = addr(x1,x2+1,x3,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q2next,pself);
            end

            if x3~=1
                ns = addr(x1,x2,x3-1,mqlength);
                [Pssa(s,ns,u),Pssa(s,s,u)] = uniformization(rate,p_q3previous,pself);
            end
            
            Pssa(s,s,u) = uniformizationself(rate,pself);
            L(s) = c'*[q1;q2;q3]/(m_rate + alpha);
        end
    end
end

end



function [p, pself] = uniformization(rate,o_p, o_pself)
global m_rate;


p = (rate/m_rate)*o_p;
pself = (rate/m_rate)*o_pself + 1 - (rate/m_rate);

end

function [pself] = uniformizationself(rate, o_pself)
global l1 l2 m1 m2 m3;
m_rate = l1+l2+m1+m2+m3;


pself = (rate/m_rate)*o_pself + 1 - (rate/m_rate);

end