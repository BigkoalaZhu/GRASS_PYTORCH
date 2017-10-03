clear all
trainingData = load('../data/trainingData_chair.mat');
data = trainingData.data;
dataNum = length(data);

maxBoxes = 30;
maxOps = 50;
maxSyms = 10;
maxDepth = 10;
copies = 1;

boxes = zeros(12, maxBoxes*dataNum*copies);
ops = zeros(maxOps,dataNum*copies);
syms = zeros(8,maxSyms*dataNum*copies);
weights = zeros(1,dataNum*copies);

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
    parent = -ones(l,1);
    while flag == 1
        flag = 0;
        for j = l:-1:1
            d = depth(j);
            if d ~= -1
                idx = treekids(j,1);
                idy = treekids(j,2);
                if idx ~= 0
                    depth(idx) = d + 1;
                    parent(idx) = j;
                end
                if idy ~= 0
                    depth(idy) = d + 1;
                    parent(idy) = j;
                end
            else
                flag = 1;
            end
        end
    end
    depthBuffer = zeros(l,1);
    flag = 1;
    while sum(depthBuffer) ~= l-1
        [~, index] = sort(depth,'descend');
        for j = 1:l
            if depthBuffer(j) == 0
                ind = j;
                break;
            end
        end
        idx = parent(ind);
        if parent(idx) ~= -1
            if treekids(parent(idx),1) ~= idx
                idy = treekids(parent(idx),1);
            end
            if treekids(parent(idx),2) ~= idx
                idy = treekids(parent(idx),2);
            end
            if idy ~= 0
                depth(idy) = depth(idy) + 0.0001*flag;
                flag = flag + 1;
                depthBuffer(idy) = 1;
            end
        end
        if idx ~= 0
            depth(idx) = depth(idx) + 0.0001*flag;
            flag = flag + 1;
        end
        depthBuffer(ind) = 1;
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
    box = repmat(box, 1, copies);
    op = repmat(op, 1, copies);
    sym = repmat(sym, 1, copies);
    boxes(:, (i-1)*maxBoxes*copies+1:i*maxBoxes*copies) = box;
    ops(:,(i-1)*copies+1:i*copies) = op;
    syms(:, (i-1)*maxSyms*copies+1:i*maxSyms*copies) = sym;
    weights(:, (i-1)*copies+1:i*copies) = b/maxBoxes;
end

save('boxes.mat','boxes');
save('ops.mat','ops');
save('syms.mat','syms');
save('weights.mat','weights');