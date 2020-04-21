function eventTimes = GetStateTimes( SessionData, eventName, isStruct)
%GETSTATETIMES(SessionData, eventName) Get per-trial event times
if nargin < 3
    isStruct = true;
end
eventTimes = [];
for i=1:SessionData.nTrials
    if isStruct
        currTrial = SessionData.states{i};
    else
        currTrial = SessionData.RawEvents.Trial{i}.States;
    end
    if isfield(currTrial, eventName)
        eventTimes = [eventTimes; getfield(currTrial, eventName)];
    end
end

end

