clear all
trainingData = load('../data/trainingData_chair.mat');
data = trainingData.data;
dataNum = length(data);

maxBoxes = 30;
maxOps = 50;
maxSyms = 10;

boxes = zeros(12, maxBoxes*dataNum);
ops = zeros(maxOps,dataNum);
syms = zeros(8,maxSyms*dataNum);

for i = 1:dataNum
    p_index = i;
    
    box = zeros(12, maxBoxes);
    op = -ones(maxOps,1);
    sym = zeros(8,maxSyms);
    
    symboxes = data{p_index}.symshapes;
    treekids = data{p_index}.treekids;
    symparams = data{p_index}.symparams;
    b = size(symboxes,2);
    l = size(treekids,1);
    opl = -ones(l,1);
    depth = -ones(l,1);
    depth(l) = 0;
    flag = 1;
    while flag == 1
        flag = 0;
        for j = l:-1:1
            d = depth(j);
            if d ~= -1
                idx = treekids(j,1);
                idy = treekids(j,2);
                if idx ~= 0
                    depth(idx) = d + 1;
                end
                if idy ~= 0
                    depth(idy) = d + 1;
                end
            else
                flag = 1;
            end
        end
    end
    for j = b+1:l
        depth(j) = depth(j) + 0.5;
    end
    [~, index] = sort(depth,'descend');
    for j = 1:l
        idx = treekids(j,1);
        idy = treekids(j,2);
        if idx == 0 && idy == 0
            opl(j) = 0;
        end
        if idx ~= 0 && idy ~= 0
            opl(j) = 1;
        end
        if idx ~= 0 && idy == 0
            opl(j) = 2;
        end    
    end
    symCount = 1;
    boxCount = 1;
    for j = 1:l
        if opl(index(j)) == 0
            op(j) = opl(index(j));
            box(:,boxCount) = symboxes(:,index(j));
            boxCount = boxCount + 1;
        end
        if opl(index(j)) == 1
            op(j) = opl(index(j));
        end
        if opl(index(j)) == 2
            op(j) = opl(index(j));
            sym(:,symCount) = symparams{1,treekids(index(j),1)}';
            symCount = symCount + 1;
        end
    end
    boxes(:, (i-1)*maxBoxes+1:i*maxBoxes) = box;
    ops(:,i) = op;
    syms(:, (i-1)*maxSyms+1:i*maxSyms) = sym;
end
save('boxes.mat','boxes');
save('ops.mat','ops');
save('syms.mat','syms');