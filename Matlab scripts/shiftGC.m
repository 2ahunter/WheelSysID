function shifted = shiftGC(gc, spokeNum)
% shifts a gain curve down to align with the spokeNum.  Note the gain curve
% is expected to be given at spoke 1
    shift = 2*spokeNum -2;
    if shift == 0 %i.e., spokeNum = 1
        shifted = gc;
    else
    n = length(gc);
    % move shifted rows from the bottom to the top and shift the rest down
    % by the shift indicated
    shifted = cat(1,gc(n-shift+1:n),gc(1:n-shift));
    end
end
