clear;clc;a=[1 2 3 4 5 6 7 8 9;2 3 4 5 6 7 8 9 10;3 4 5 6 7 8 9 10 11;4 5 6 7 8 9 10 11 12]sigma = a'*a/4[u,s,v]=svd(sigma)ur = u(:,1:2)z=a*urz*ur'