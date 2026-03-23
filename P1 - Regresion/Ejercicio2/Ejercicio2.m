%{
Ejercicio 2: Regresión lineal - GLM

    a) Leer los datos de la base de datos de calidad del vino tinto almacenados
    en la base de datos "Wine Quality".

    b) Resolver el problema de regresión GLM que estima el valor de la calidad
    del vino en función del resto de parámetros.

    c) Calcular el error medio en la estimación de la calidad del vino que se
    comete sobre los datos disponibles.
%}


red_wine = table2array(readtable('winequality-red.csv'));

quality = red_wine(:,12);    % Salida deseada

% Entrenamiento
A = [red_wine(:, 1:11) ones(height(red_wine),1)];
coefs = pinv(A)*quality;

% Test con los datos disponibles
quality_estim = round(A*coefs);
% Redondeamos los datos para que conserven su naturaleza discreta

r = quality - quality_estim;
E = (r'*r)/length(red_wine);





% Dibujo del resultado
plot(quality,'.r');hold on;
plot(quality_estim,'.w');hold off
disp (E)