clear all
%close all
clc


[x1,y1,z1,e1]=openTraFileSingle('Crystal1_200keV.inc1.id1.tra');


%%
c(:,1)=x1;
c(:,2)=y1;
edges = {(-3:0.05:3),(-3:0.05:3)}; 
PM=hist3(c, 'Edges', edges);

figure()
p1=pcolor((-3:0.05:3),(-3:0.05:3),PM);


%%
S=sum(PM,2);
x=-3:0.05:3;

figure()
plot(x,S)


% Fit model to data.
[xData, yData] = prepareCurveData( x, S );
ft = fittype( 'gauss1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0];
opts.StartPoint = [57000 0 0.3];
[fitresult, gof] = fit( xData, yData, ft, opts  );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData );

FWHM = 2*sqrt(log(2)) * fitresult.c1

%%
figure()
histogram(y2,(-4:0.05:4))
figure()
histogram(z2,(-20:0.05:0))