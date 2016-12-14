clear
M_full = load('hsi-2015.csv');

% ,Adj_Close,Close,Date,High,Low,Open,Symbol,Volume
% 1,2       ,3    ,4   ,5   ,6  ,7   ,8     ,9

% REMINDER==== May add Scaling into it later

M = M_full(1:2000,:);

window =100;

for i=1:length(M)-window % number of samples
    % determine if it increased or decrease  
    if ((M(i+window,3)-M(i+window,7)) >= 0 )
        Y(i,1) = 1;
    else
        Y(i,1) = -1;
    end
    for j=1:2:window
        for k=0:1
            if k == 0
                X(i,j+k) = M(i+j+k-1,7);
%                 disp(j+k);
            else
                X(i,j+k) = M(i+j+k-1,3);
%                 disp(j+k);
            end
        end
    end
    
end

b = regress(Y,X);

M = M_full(2001:3727,:) ;
Y = [];
X = [];

for i=1:length(M)-window % number of samples
    % determine if it increased or decrease  
    if ((M(i+window,3)-M(i+window,7)) >= 0 )
        Y(i,1) = 1;
    else
        Y(i,1) = -1;
    end
    for j=1:2:window
        for k=0:1
            if k == 0
                X(i,j+k) = M(i+j+k-1,7);
%                 disp(j+k);
            else
                X(i,j+k) = M(i+j+k-1,3);
%                 disp(j+k);
            end
        end
    end
end

acc = 0

for i=1:length(M)-window
    guess(i,1) = X(i,:)*b;
    guess(i,2) = Y(i);
    if guess(i,1) >= 0
        tmp_x =1;
    else
        tmp_x =-1;
    end
    % calculate accurate count
    if Y(i) == tmp_x
        tmp_acc = 1;
    else
        tmp_acc = 0;
    end
    acc = acc + tmp_acc;
end

accurancy = acc / length(Y);

% cross validation
g_pt = 1;

y_guess = X(g_pt,:)*b

y_real = Y(g_pt)


plot(window,accurancy)



