function gabor = GaborFilterBank(scales,orientations,m,n)
%GABORFILTERBANK Summary of this function goes here
if (nargin~=4)
    error('There must be four input arguments')
end
%create Gabor filters
gabor=cell(scales,orientations);
freq=0.25;
a=sqrt(2);
b=sqrt(2);

for i=1:scales
    f=freq/((sqrt(2))^(i-1));
    alpha=f/a;
    beta=f/b;
    
    for j=1:orientations
        theta=((j-1)/orientations)*pi;
        Filter=zeros(m,n);
        
        for x=1:m
            for y=1:n
                A=(x-((m+1)/2))*cos(theta)+(y-((n+1)/2))*sin(theta);
                B=-(x-((m+1)/2))*sin(theta)+(y-((n+1)/2))*cos(theta);
                Filter(x,y)=(f^2/(pi*a*b))*exp(-((alpha^2)*(A^2)+(beta^2)*(B^2)))*exp(1i*2*pi*f*A);
            end
        end
        gabor{i,j}=Filter;
    end
end

%Real parts of Gabor filters
figure('NumberTitle','Off','Name','Gabor filters with orientations');
for i=1:scales
    for j=1:orientations
        subplot(scales,orientations,(i-1)*orientations+j);
        imshow(real(gabor{i,j}),[])
    end
end

%Magnitudes of Gabor filters
figure('NumberTitle','Off','Name','Real part of Gabor filters');
for i=1:scales
    for j=1:orientations
        subplot(scales,orientations,(i-1)*orientations+j);
        imshow(abs(gabor{i,j}),[])
    end
end
end

