function shifted = shiftGC(gc, spokeNum)
    shift = 2*spokeNum -2;
    if shift == 0
        shifted = gc;
    else
    n = length(gc);
    shifted = cat(1,gc(n-shift+1:n),gc(1:n-shift));
    end
end
