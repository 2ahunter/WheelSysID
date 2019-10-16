function shifted = shiftTC(tc, shift)
% forward circular shift a vector, tc, by the number of elements in shift
    if shift <0
        shift = 0;
    else
        shift = rem(shift, length(tc));
    end
    if shift == 0
        shifted = tc;
    else
    n = length(tc);
    shifted = cat(1,tc(n-shift+1:n),tc(1:n-shift));
    end
end