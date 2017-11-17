%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Testando resize da imagem

clear; clc; close all;

ratio = zeros(1, 40);
mini  = zeros(1, 40);
maxi  = zeros(1, 40);

fonte = 'C:\Users\vinic\Desktop\poste\';
destino = [fonte, 'res\'];
mkdir(destino)
n = size(dir([fonte, '*.PNG']), 1);

for i=1:n
    nome = [fonte, 'poste', num2str(i), '.PNG'];
    original = imread(nome);
    
    if i==13
        continue
    end
%     imshow(original)
    
    ratio(i) = size(original, 1)/size(original, 2);
    maxi(i)  = size(original, 1);
    mini(i)  = size(original, 2);

    res = imresize(original, [380 130]);
    imwrite(res, [destino, 'poste', num2str(i), '.jpg']);
    
    imshow(res)
    
    pause(1)

end
r  = mean(ratio);
ma = mean(maxi );
mi = mean(mini );
disp(r)
disp(ma)
disp(mi)
