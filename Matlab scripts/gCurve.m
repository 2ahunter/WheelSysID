function gainCurve = gCurve(data, spoke)    
    n = length(data);
    index = spoke*2 -1;
    % shift data so that the spoke causing the influence curve is at first
    % data point
    shiftLeft = index-1;
    gainCurve = cat(1,data(shiftLeft+1:n),data(1:shiftLeft));
    
end