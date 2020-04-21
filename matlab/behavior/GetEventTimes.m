function eventTimes = GetEventTimes( SessionData, eventName, isStruct)
%GETSTATETIMES(SessionData, eventName) Get per-trial event times
if nargin < 3
    isStruct = true;
end
eventTimes = {};
for i=1:SessionData.nTrials
    if isStruct
        currTrial = SessionData.events{i};
    else
        currTrial = SessionData.RawEvents.Trial{i}.Events;
    end
    if isfield(currTrial, eventName)
        eventTimes{end+1} = getfield(currTrial, eventName);
    else
        eventTimes{end+1} = [];
    end
end

end

