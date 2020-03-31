clear ; close all; clcx=-10:0.1:10;lengthx=length(x);y=zeros(lengthx,1);%for i=1:lengthx%  y(i)=1/(1+e^(-1*x(i)));
%endfory=1./(1+exp(-x));plot(x,y);set(gca,'YAxisLocation','origin');