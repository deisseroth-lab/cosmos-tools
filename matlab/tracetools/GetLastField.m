function  field = GetLastField(fields, matchString, isDate)
%%% fields is a struct of strings (i.e. from calling fieldnames() on
%%% another struct). 
%%% matchString is a string that specifies a subset of the field in the
%%% struct.
%%% isDate: If true,  specifies that the field name are of the form
%%% {'name_01_Jan_HH_MM_SS}. In this case, cannot just take the
%%% alphetically last string, must account for the fact that months have an
%%% ordering. 
%%% field is the 'most recent', (i.e. alphabetically last field name) of
%%% that subset. 
%%% THIS MAY FAIL IF YOU ARE PROCESSING PAST MIDNIGHT WHEN STARTING A NEW
%%% MONTH.....THIS IS BAD DESIGN OF A NAMING SCHEME...

if ~exist('isDate', 'var') || isempty(isDate)
    isDate = false;
end

subfields = fields(contains(fields, matchString));

sorted = sort(subfields);

MONTHS = {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'};
if isDate
    months = [];
    for i = 1:numel(sorted)
        ss = strsplit(sorted{i}, '_');
        months(i) = find(contains(MONTHS, ss{3}));        
    end
    [sortedMonths, inds] = sort(months, 'ascend');
    if ~isempty(find(months == 1)) & ~isempty(find(months == 12))
        month = MONTHS{1};
    else
        month = MONTHS{sortedMonths(end)};
    end
    
    subsubfields = subfields(contains(subfields, month));
    sorted = sort(subsubfields);
    field = sorted{end};
else
    field = sorted{end};
end

