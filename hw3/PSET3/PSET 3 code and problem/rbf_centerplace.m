clear all;
%load rbf_c_store2;
n2_maxcenter=10;
amin=0;
amax=5;
for ii=1:n2_maxcenter
lincenter(1,ii)=amin+(ii-1)*(amax-amin)/(n2_maxcenter-1);
lincenter2(1,ii)=amax-(ii-1)*(amax-amin)/(n2_maxcenter-1);
end
rbf_c=[lincenter lincenter2;lincenter lincenter]

% save rbf_c_store10 rbf_c;
% 
%%
n2_maxcenter=20;
amin=0;
amax=5;
umin=1;
umax=5;
for ii=1:n2_maxcenter
lincenter(1,ii)=amin+(ii-1)*(amax-amin)/(n2_maxcenter-1);
lincenter2(1,ii)=amax-(ii-1)*(amax-amin)/(n2_maxcenter-1);
lincenter3(1,ii)=umin+(ii-1)*(umax-umin)/(n2_maxcenter-1);
linecenter4(1,ii)=umax-(ii-1)*(umax-umin)/(n2_maxcenter-1);
end
rbf_c=[lincenter lincenter2;lincenter lincenter; lincenter3 linecenter4]


%% discrete placed at discrete states
clear all
n2_maxcenter=50;
amin=0;
amax=5;
umin=1;
umax=5;
rbf_c=zeros(3,n2_maxcenter);
rbf_c(1:2,1:n2_maxcenter)=randi(amax,2,n2_maxcenter)
rbf_c(3,1:n2_maxcenter)=randi(umax,1,n2_maxcenter)


%% uniformly distributed over the reals
clear all
n2_maxcenter=40;
amin=0;
amax=5;
umin=1;
umax=5;
rbf_c=zeros(3,n2_maxcenter);
rbf_c(1:2,1:n2_maxcenter)=amin+(amax-amin).*rand(2,n2_maxcenter)
rbf_c(3,1:n2_maxcenter)=umin+(umax-umin).*rand(1,n2_maxcenter)
%
% counter=1;
% rbf_c=zeros(2,25);
% count=0
% for kk=1:5
% for jj=1:5
% rbf_c(1,jj+count)=counter;
% rbf_c(2,jj+count)=jj;
% end
% count=count+5;
% counter=counter+1;
% end
% 
% figure
% 

plot(rbf_c(1,:),rbf_c(2,:),'*')

plot3(rbf_c(1,:),rbf_c(2,:),rbf_c(3,:),'*')

