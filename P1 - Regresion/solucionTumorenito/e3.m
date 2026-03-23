% Cargar BD
dataset = readtable('COVID-19.csv');

% Obtener los datos de China y España
[yC, x1, x2, x3] = getdata(dataset, 'China', 1/147);    % Datos de China
[yE, x1E, x2E, x3E] = getdata(dataset, 'Spain', 1/93);  % Datos de España
n = length(yC);

% Cálculo 
A = [x1, x2, x3, ones(n, 1)];    % Matriz de China
coefs = pinv(A) * yC;            % Coeficientes

E = [x1E, x2E, x3E, ones(n, 1)]; % Creo la matriz de España
yr = E * coefs;                  % Saco la y estimada para España

% Dibujo
figure;
xr = 1:n;
plot(xr, yE, '-r'); hold on;  % Datos reales de España
plot(xr, yr, '-g');           % Predicción para España
plot(xr, yC, '-b'); hold off; % Datos de China como referencia

legend('España (Real)', 'España (Predicción)', 'China (Referencia)');
xlabel('Días desde el 22 de febrero de 2020');
ylabel('Casos activos normalizados');
title('Predicción de la progresión del COVID-19 en España');
grid on;