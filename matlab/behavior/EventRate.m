function [rate, timeStamps] = EventRate( licks, binSize, maxTime )
% EVENTRATE(licks, binSize, maxTime)
% compute lick rate for cell array of event time vectors
% called licks in this case because used to compute lick rate...

% rate(1) is the rate of licks in the bin from time = 0 to time = +binSize

rate = zeros(round(maxTime/binSize), numel(licks));
for i=1:numel(licks)
    currLicks = licks{i};
    k = 1;
    for j=0:binSize:maxTime-binSize        
        currWin = [j j+binSize];
        rate(k,i) = sum(currLicks >= currWin(1) & currLicks < currWin(2))/binSize;
        k = k + 1;
    end
    timeStamps = [0:binSize:maxTime - binSize];
end

end

