

function [xs,ys]=getsnake(image)

hold on;
xs=[];
ys=[];

but=1;
hold on;

xy=[];
n=0;
disp('left mouse button picks points');
disp('Right mouse button picks end');
but=1;
while but==1
    [xi,yi,but]=ginput(1);
    plot(xi,yi,'ro')
    n=n+1;
    xy(:,n)=[xi;yi];
    
end

n=n+1;
xy(:,n)=[xy(1,1);xy(2,1)];

t=1:n;
ts=1:0.1:n;
xys=spline(t,xy,ts);

xs=xys(1,:);
ys=xys(2,:);

