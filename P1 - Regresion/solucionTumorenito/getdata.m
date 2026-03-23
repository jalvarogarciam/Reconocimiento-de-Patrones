function [active_res, confirmed_res, death_res, recovered_res, date] = getdata(data,country,normalization)
data.Country_Region = categorical(data.Country_Region);
data.Case_Type = categorical(data.Case_Type);

if nargin>1
    data = data(data.Country_Region==country,:);
end 

if nargin < 3
    normalization = 1;
end

active = data(data.Case_Type=='Active',:);
confirmed = data(data.Case_Type=='Confirmed',:);
death = data(data.Case_Type=='Deaths',:);
recovered = data(data.Case_Type=='Recovered',:);

active_res = splitapply(@sum,active.Cases,findgroups(active.Date))*normalization;
confirmed_res = splitapply(@sum,confirmed.Cases,findgroups(confirmed.Date))*normalization;
death_res = splitapply(@sum,death.Cases,findgroups(death.Date))*normalization;
recovered_res = splitapply(@sum,recovered.Cases,findgroups(recovered.Date))*normalization;

date = unique(active.Date);

% figure
% plot(date, active_res),hold on
% plot(date, confirmed_res)
% plot(date, death_res)
% plot(date, recovered_res), hold off
% legend('active','confirmed','death','recovered')