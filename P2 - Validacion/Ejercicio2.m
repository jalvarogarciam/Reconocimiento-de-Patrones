%{
Ejercicio 2: ¿Ha funcionado nuestra estrategia?

Para comprobar si ha funcionado nuestra estrategia, haremos un pequeño truco, dado que
sabemos de dónde proceden los datos x e y. Genere 1.000.000 de datos, y calcule el error
cometido por cada modelo.
    x = rand(1,1000000);
    y = exp(x.^3 - x.^2 + 0.01*x + 2) + 0.04 * randn(size(x));
Este error es más fiable, obviamente que el obtenido usando CV10. Aún así, ¿son
coherentes ambos resultados?

%}

clc, clear all, close all;



%% ---1. Entrenamos los modelos con los datos de entrenamiento

% Generamos los datos de entrenamiento (100 datos)
rand('seed', 0);
randn('seed', 0);
xtrn = rand(1,100);
ytrn = exp(xtrn.^3 - xtrn.^2 + 0.01*xtrn + 2) + 0.04 * randn(size(xtrn));


% 1 -> y = a + bx
coefs1 = polyfit(xtrn, ytrn, 1);

% 2 -> y = a + bx + cx^2
coefs2 = polyfit(xtrn, ytrn, 2);

% 3 -> y = a + bx + cx^2 + dx^3
coefs3 = polyfit(xtrn, ytrn, 3);

% 4 -> y = a + bx + cx^2 + dx^3 + esin(x)
A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', (sin(xtrn))', ones(length(xtrn), 1)];
coefs4 = pinv(A) * ytrn';

% 5 -> y = a + bx + cx^2 + dx^3 + esin(x) + fcos(x)
A = [(xtrn.^3)', (xtrn.^2)', (xtrn)', (sin(xtrn))', (cos(xtrn))', ones(length(xtrn), 1)];
coefs5 = pinv(A) * ytrn';




%% ---2. Testeamos los datos con el millón de datos y calculamos los errores.
rand('seed', 0);
randn('seed', 0);
xtst = rand(1,1000000);
ytst = exp(xtst.^3 - xtst.^2 + 0.01*xtst + 2) + 0.04 * randn(size(xtst));

yestim = zeros(5, length(ytst));
errores = zeros(1,5);


% 1 -> y = a + bx
yestim(1,:) = polyval(coefs1, xtst);
errores(1) = mean((ytst-yestim(1,:)).^2);


% 2 -> y = a + bx + cx^2
yestim(2,:) = polyval(coefs2, xtst);
errores(2) = mean((ytst-yestim(2,:)).^2);


% 3 -> y = a + bx + cx^2 + dx^3
yestim(3,:) = polyval(coefs3, xtst);
errores(3) = mean((ytst-yestim(3,:)).^2);


% 4 -> y = a + bx + cx^2 + dx^3 + esin(x)
yestim(4,:) = coefs4(1)*xtst.^3 + coefs4(2)*xtst.^2 + coefs4(3)*xtst + coefs4(4)*sin(xtst) + coefs4(5);
errores(4) = mean((ytst-yestim(4,:)).^2);


% 5 -> y = a + bx + cx^2 + dx^3 + esin(x) + fcos(x)
yestim(5,:) = coefs5(1)*xtst.^3 + coefs5(2)*xtst.^2 + coefs5(3)*xtst + coefs5(4)*sin(xtst) + coefs5(5)*cos(xtst)+ coefs5(6);
errores(5) = mean((ytst-yestim(5,:)).^2);


%% ---3. Mostramos los resultados
fprintf('Error Real Modelo 1: %f\n', errores(1));
fprintf('Error Real Modelo 2: %f\n', errores(2));
fprintf('Error Real Modelo 3: %f\n', errores(3));
fprintf('Error Real Modelo 4: %f\n', errores(4));
fprintf('Error Real Modelo 5: %f\n', errores(5));

[err_min, mejor_mod] = min(errores);
fprintf('\n=> EN EL MUNDO REAL (1M datos), EL MEJOR MODELO ES EL %d\n', mejor_mod);







%% ---4. Dibujamos una muestra de los datos y los 5 modelos
disp('Generando gráfica comparativa...');

% Creamos un vector ordenado para que las curvas salgan perfectas.
x_plot = linspace(min(xtst), max(xtst), 10000); 

% Calculamos las estimaciones para ese vector ordenado
y_plot1 = polyval(coefs1, x_plot);
y_plot2 = polyval(coefs2, x_plot);
y_plot3 = polyval(coefs3, x_plot);
y_plot4 = coefs4(1)*x_plot.^3 + coefs4(2)*x_plot.^2 + coefs4(3)*x_plot + coefs4(4)*sin(x_plot) + coefs4(5);
y_plot5 = coefs5(1)*x_plot.^3 + coefs5(2)*x_plot.^2 + coefs5(3)*x_plot + coefs5(4)*sin(x_plot) + coefs5(5)*cos(x_plot)+ coefs5(6);

% Dibujamos
figure;
idx = randperm(numel(xtst), 10000);
plot(xtst(idx), ytst(idx), 'wo', 'MarkerFaceColor', 'k', 'MarkerSize', 2); 
hold on;

plot(x_plot, y_plot1, 'LineWidth', 1.5);                      % Modelo 1 (Recta)
plot(x_plot, y_plot2, 'LineWidth', 1.5);                      % Modelo 2 (Parábola)
plot(x_plot, y_plot3, 'g', 'LineWidth', 3);                   % Modelo 3 (El ganador real)
plot(x_plot, y_plot4, '--c', 'LineWidth', 1.5);               % Modelo 4
plot(x_plot, y_plot5, ':r', 'LineWidth', 1.5);                % Modelo 5

title('Comparación de los 5 Modelos sobre los 100 datos originales');
xlabel('X'); ylabel('Y');
legend('Datos Reales (100 pts)', 'Mod 1 (Recta)', 'Mod 2 (Parábola)', ...
       'Mod 3 (Grado 3 - Óptimo)', 'Mod 4 (+seno)', 'Mod 5 (+seno+cos)', ...
       'Location', 'best');
axis([min(xtst) max(xtst) min(ytst)-0.1 max(ytst)+0.1]);
grid on;
hold off;











%{
SALIDA:

Error Real Modelo 1: 0.084217
Error Real Modelo 2: 0.017621
Error Real Modelo 3: 0.001648
Error Real Modelo 4: 0.001648
Error Real Modelo 5: 0.001647

=> EN EL MUNDO REAL (1M datos), EL MEJOR MODELO ES EL 5


Como podemo observar, los 3 últimos modelos obtienen casi el mismo error,
lo que tiene bastante sentido, ya que son modelos polinómicos de grado 3, 
 y sabemos que los datos se han generado de dicha forma (en los 2 últimos 
hay sumandos trigonométricos, pero no influyen mucho). 
%}